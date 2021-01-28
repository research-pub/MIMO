from .modeling import BertEncoder, AttentionPooling, BertLayerNorm, PositionEmbeddings, BertEncoderDag
import torch.nn as nn
import torch
import os
import logging

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER
logger = logging.getLogger(__name__)


class DAGAttention2D(nn.Module):
    def __init__(self, in_features, attention_dim_size):
        super(DAGAttention2D, self).__init__()
        self.attention_dim_size = attention_dim_size
        self.in_features = in_features
        self.linear1 = nn.Linear(in_features, attention_dim_size)
        self.linear2 = nn.Linear(attention_dim_size, 1)

    def forward(self, leaves, ancestors, mask=None):
        # concatenate the leaves and ancestors
        mask = mask.unsqueeze(2)

        x = torch.cat((leaves * mask, ancestors * mask), dim=-1)

        # Linear layer
        x = self.linear1(x)

        # tanh activation
        # x = torch.tanh(x)
        x = torch.relu(x)

        # linear layer
        # x = self.linear2(x * mask)
        x = self.linear2(x)

        mask_attn = (1.0 - mask) * VERY_NEGATIVE_NUMBER
        x = x + mask_attn

        # softmax activation
        x = torch.softmax(x, dim=1)

        # weighted sum on ancestors
        x = (x * ancestors * mask).sum(dim=1)
        return x


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pre-trained models.
    """
    def __init__(self, config,  *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, config, state_dict=None, *inputs, **kwargs):
        print('parameters in inputs: ', *inputs)

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)

        if state_dict is None:
            weights_path = os.path.join(pretrained_model_name, 'pytorch_model.bin')
            state_dict = torch.load(weights_path).state_dict()

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))

        return model, missing_keys


class KnowledgeEncoder(nn.Module):
    def __init__(self, config):
        super(KnowledgeEncoder, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.embed_dag = None

        self.encoder_knowledge = BertEncoderDag(config)

        self.pooling_visit = AttentionPooling(config)
        self.pooling_dag = AttentionPooling(config)

        self.embed_init = nn.Embedding(config.num_tree_nodes, config.hidden_size)
        self.embed_inputs = nn.Embedding(config.code_size, self.hidden_size)

        self.dag_attention = DAGAttention2D(2 * config.hidden_size, config.hidden_size)

    def forward(self, input_ids, code_mask=None, output_attentions=False):

        # for knowledge graph embedding
        leaves_emb = self.embed_init(self.config.leaves_list)
        ancestors_emb = self.embed_init(self.config.ancestors_list)
        dag_emb = self.dag_attention(leaves_emb, ancestors_emb, self.config.masks_list)
        padding = torch.zeros([1, self.hidden_size], dtype=torch.float32).to(self.config.device)
        dict_matrix = torch.cat([padding, dag_emb], dim=0)
        self.embed_dag = nn.Embedding.from_pretrained(dict_matrix, freeze=False)

        # inputs embedding
        input_tensor = self.embed_inputs(input_ids)  # bs, visit_len, code_len, embedding_dim
        input_shape = input_tensor.shape
        inputs = input_tensor.view(-1, input_shape[2], input_shape[3])  # bs * visit_len, code_len, embedding_dim

        # entity embedding
        input_tensor_dag = self.embed_dag(input_ids)
        # bs * visit_len, code_len, embedding_dim
        inputs_dag = input_tensor_dag.view(-1, input_shape[2], input_shape[3])

        inputs_mask = code_mask.view(-1, input_shape[2])  # bs * visit_len, code_len

        # attention mask for encoder
        extended_attention_mask = inputs_mask.unsqueeze(1).unsqueeze(2)  # bs * visit_len,1,1 code_len
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * VERY_NEGATIVE_NUMBER
        # knowledge encoder
        visit_outputs, dag_outputs, all_attentions = self.encoder_knowledge(inputs, extended_attention_mask,
                                                            inputs_dag, output_all_encoded_layers=False,
                                                            output_attentions=output_attentions)

        # attention mask for pooling
        attention_mask = inputs_mask.unsqueeze(2)  # bs * visit_len,code_len,1
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * VERY_NEGATIVE_NUMBER

        # visit attention pooling
        visit_pooling = self.pooling_visit(visit_outputs[-1], attention_mask)
        visit_outs = visit_pooling.view(-1, input_shape[1], input_shape[3])  # bs, visit_len, embedding_dim

        # knowledge attention pooling
        dag_pooling = self.pooling_dag(dag_outputs[-1], attention_mask)
        dag_outs = dag_pooling.view(-1, input_shape[1], input_shape[3])  # bs, visit_len, embedding_dim

        return visit_outs, dag_outs, all_attentions


class NextDxPrediction(PreTrainedBertModel):
    def __init__(self, config):
        super(NextDxPrediction, self).__init__(config)

        self.encoder = KnowledgeEncoder(config)

        self.encoder_patient = BertEncoder(config)
        self.position_embedding = PositionEmbeddings(config)

        self.classifier_patient = nn.Linear(config.hidden_size, config.num_ccs_classes)
        self.classifier_entity = nn.Linear(config.hidden_size, config.num_visit_classes)

        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                visit_mask=None,
                code_mask=None,
                labels_visit=None,
                labels_entity=None,
                output_attentions=False
                ):

        lengths = visit_mask.sum(axis=-1)
        outputs_visit, outputs_entity, code_attentions = self.encoder(input_ids, code_mask, output_attentions)

        # add position embedding
        visit_outs = self.position_embedding(outputs_visit)

        extended_attention_mask = visit_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * VERY_NEGATIVE_NUMBER
        patient_outputs, visit_attentions = self.encoder_patient(visit_outs, extended_attention_mask,
                                                                 output_all_encoded_layers=False,
                                                                 output_attentions = output_attentions)

        prediction_scores_patient = self.classifier_patient(patient_outputs[-1])
        prediction_scores_patient = torch.sigmoid(prediction_scores_patient)

        prediction_scores_entity = self.classifier_entity(outputs_entity)
        prediction_scores_entity = torch.sigmoid(prediction_scores_entity)

        if labels_visit is not None and labels_entity is not None:
            logEps = 1e-8
            cross_entropy_patient = -(labels_visit * torch.log(prediction_scores_patient + logEps) +
                              (1. - labels_visit) * torch.log(1. - prediction_scores_patient + logEps))
            likelihood_patient = cross_entropy_patient.sum(axis=2).sum(axis=1) / lengths
            loss_patient = torch.mean(likelihood_patient)

            cross_entropy_entity = -(labels_entity * torch.log(prediction_scores_entity + logEps) +
                              (1. - labels_entity) * torch.log(1. - prediction_scores_entity + logEps))
            likelihood_entity = cross_entropy_entity.sum(axis=2).sum(axis=1) / lengths
            loss_entity = torch.mean(likelihood_entity)

            total_loss = loss_patient + self.config.lamda * loss_entity
            return total_loss
        else:
            return prediction_scores_patient, code_attentions, visit_attentions

