import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List
from typing import Union

import torch
from sklearn.metrics import f1_score, classification_report
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import Trainer, RobertaForMaskedLM, PreTrainedModel, BertForMaskedLM, \
    AutoModelForMaskedLM, PretrainedConfig, BertConfig
from transformers import TrainingArguments, AutoTokenizer, \
    EvalPrediction
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertPooler, BertModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from stancedatasets import StanceDataset


@dataclass
class StanceClassifierOutput(SequenceClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    scl_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    masked_probas: Optional[Tuple[torch.FloatTensor]] = None


class SCLoss(nn.Module):
    """
    own implementation of Gunel et al. 2020
    Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning
    https://doi.org/10.48550/arXiv.2011.01403
    """
    def __init__(self, temperature: float = 0.3, mean_reduction=False):
        super(SCLoss, self).__init__()
        self.temperature = temperature
        self.mean_reduction = mean_reduction

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        device = features.device
        features = torch.nn.functional.normalize(features, dim=1)
        loss = torch.tensor(0.0).to(device)
        loss_ni = torch.tensor(0.0).to(device)
        for i in range(features.shape[0]):
            mask_same = torch.ones_like(labels).bool().to(device)
            mask_same[i] = 0
            mask_same_label = labels[i] == labels
            mask_same_label[i] = 0
            if mask_same_label.sum() == 0:
                continue
            pos_term = torch.exp(features[mask_same_label].mm(features[i].unsqueeze(1)).squeeze() / self.temperature)
            other_term = torch.exp(features[mask_same].mm(features[i].unsqueeze(1)).squeeze() / self.temperature).sum()
            loss_i = torch.log(pos_term / other_term)
            loss -= loss_i.mean()
            loss_ni += 1.0

        if self.mean_reduction:
            loss = loss / loss_ni

        return loss


class StanceLoss(nn.Module):
    """
    stance contrastive loss from JointCL
    DOI: 10.18653/v1/2022.acl-long.7
    https://github.com/HITSZ-HLT/JointCL
    """
    def __init__(self, temperature, contrast_mode='all',
                 base_temperature=0.07):
        super(StanceLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        features = features.unsqueeze(1)
        features = torch.nn.functional.normalize(features, dim=2)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().add(0.0000001).to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        mask = mask.repeat(anchor_count, contrast_count)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask)-mask) * logits_mask

        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)

        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))

        return loss


def calc_weighted_cross_entropy(labels, logits, num_labels, sqrt=False):
    """
    calculates weighted cross entropy loss
    use inverse of class frequency as weight, or sqrt of inverse if sqrt=True
    """
    weights = torch.Tensor([1.0 / (labels == 0).sum(),
                            1.0 / (labels == 1).sum(),
                            1.0 / (labels == 2).sum()]).nan_to_num(0, 0, 0).to(logits.device)
    if sqrt:
        weights = torch.sqrt(weights)
    weights = weights / weights.sum()
    loss_fct = CrossEntropyLoss(weight=weights)
    return loss_fct(logits.view(-1, num_labels), labels.view(-1))


class StanceEncoderModel(PreTrainedModel):

    config_class = BertConfig
    logger = logging.getLogger("StanceEncoderModel")

    @staticmethod
    def instantiate_from_base(base_model_name_or_path: str, task_specific_params: dict):
        base_model = AutoModelForMaskedLM.from_pretrained(base_model_name_or_path,
                                                          task_specific_params=task_specific_params)
        if isinstance(base_model, BertForMaskedLM):
            base_model.config.base_model_type = 'bert'
            instance = StanceEncoderModel(base_model.config, load_base_models=False)
            instance.base_enc_model = base_model.bert
            instance.lm_head = base_model.cls
            instance.post_init()
        elif isinstance(base_model, RobertaForMaskedLM):
            base_model.config.base_model_type = 'roberta'
            instance = StanceEncoderModel(base_model.config, load_base_models=False)
            instance.base_enc_model = base_model.roberta
            instance.lm_head = base_model.lm_head
            instance.post_init()
        else:
            raise ValueError(f'{base_model.__class__} is not supported')

        return instance

    def __init__(self, config, load_base_models=True):
        super().__init__(config)
        task_specific_params = config.task_specific_params
        self.num_labels = task_specific_params.get('num_labels', 3)
        self.mask_token_id = task_specific_params['mask_token_id']
        self.verbalizer_token_ids = task_specific_params['verbalizer_token_ids']
        self.clf_hidden_dim = task_specific_params.get('clf_hidden_dim', 300)
        self.clf_drop_prob = task_specific_params.get('clf_drop_prob', 0.2)
        self.clf_small_head = task_specific_params.get('clf_small_head', False)
        self.clf_gelu_head = task_specific_params.get('clf_gelu_head', False)
        self.clf_roberta_head = task_specific_params.get('clf_roberta_head', False)
        self.use_cls_token = task_specific_params.get('use_cls_token', False)
        self.use_second_cls_token = task_specific_params.get('use_second_cls_token', False)
        self.custom_cls_tokens = task_specific_params.get('custom_cls_tokens', [])
        self.masked_lm = task_specific_params.get('masked_lm', True)
        self.masked_lm_n_tokens = task_specific_params.get('masked_lm_tokens', 1)
        self.masked_lm_new_token = task_specific_params.get('masked_lm_new_token', None)
        self.return_masked_probas = task_specific_params.get('return_masked_probas', False)
        self.masked_lm_verbalizer = task_specific_params.get('masked_lm_verbalizer', False)
        self.masked_lm_like_clf_head = task_specific_params.get('masked_lm_like_clf_head', False)
        self.weighted_loss = task_specific_params.get('weighted_loss', False)
        self.weighted_loss_sqrt = task_specific_params.get('weighted_loss_sqrt', False)
        self.class_weights = task_specific_params.get('class_weights', None)
        self.class_weights = torch.Tensor(self.class_weights) if self.class_weights is not None else None
        self.contrastive_loss = task_specific_params.get('contrastive_loss', False)
        self.contrastive_loss_v2 = task_specific_params.get('contrastive_loss_v2', False)
        self.contrastive_loss_v3 = task_specific_params.get('contrastive_loss_v3', False)
        self.contrastive_temp = task_specific_params.get('contrastive_temperature', 0.3)
        self.contrastive_lambda = task_specific_params.get('contrastive_lambda', 0.9)
        self.use_bert_pooling_out = task_specific_params.get('use_bert_pooling_out', False)

        if load_base_models:
            if config.base_model_type == 'bert':
                base_model = BertForMaskedLM(config)
                self.base_enc_model = base_model.bert
                self.lm_head = base_model.cls
            elif config.base_model_type == 'roberta':
                base_model = RobertaForMaskedLM(config)
                self.base_enc_model = base_model.roberta
                self.lm_head = base_model.lm_head
            else:
                raise ValueError(f'{config.base_model_type} is not supported')

        if self.masked_lm:
            if self.mask_token_id is None:
                raise ValueError('mask_token_id must be provided if masked_lm is True')
            else:
                self.mask_token_id = self.mask_token_id

        if self.masked_lm_verbalizer:
            if self.masked_lm is not True:
                raise ValueError('masked_lm_verbalizer masked_lm to be True')
            if self.verbalizer_token_ids is None:
                raise ValueError('verbalizer_token_ids must be provided if masked_lm_verbalizer is True')
            else:
                self.verbalizer_token_ids = self.verbalizer_token_ids
                assert self.masked_lm_n_tokens == 1, 'masked_lm_tokens must be 1 if masked_lm_verbalizer is True'

        if self.masked_lm_new_token:
            if self.masked_lm is not True:
                raise ValueError('masked_lm_new_token_requires masked_lm to be True')

        if self.masked_lm_verbalizer and (self.contrastive_loss or self.contrastive_loss_v2 or self.contrastive_loss_v3):
            raise ValueError('masked_lm_verbalizer is incompatible with contrastive_loss')

        if sum([self.contrastive_loss, self.contrastive_loss_v2, self.contrastive_loss_v3]) > 1:
            raise ValueError('only one contrastive loss can be used at a time')

        scl_class = StanceLoss if self.contrastive_loss else SCLoss
        if self.contrastive_loss or self.contrastive_loss_v2:
            self.logger.info(f'using contrastive loss: {scl_class.__name__}')
            self.stance_loss = scl_class(self.contrastive_temp)
        elif self.contrastive_loss_v3:
            self.logger.info(f'using contrastive loss: {scl_class.__name__} with mean reduction')
            self.stance_loss = scl_class(self.contrastive_temp, mean_reduction=True)

    def post_init(self):

        hidden_size_multiplier = 0
        if self.use_cls_token:
            hidden_size_multiplier += 1
        if self.use_second_cls_token:
            hidden_size_multiplier += 1
        if self.custom_cls_tokens is not None:
            hidden_size_multiplier += len(self.custom_cls_tokens)
        if self.masked_lm:
            hidden_size_multiplier += self.masked_lm_n_tokens

        self.logger.info(f'representation size: {self.config.hidden_size * hidden_size_multiplier}')
        if not self.masked_lm_verbalizer:
            if self.clf_small_head:
                self.logger.info('using small classifier head')
                self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(self.clf_drop_prob),
                    torch.nn.Linear(self.config.hidden_size*hidden_size_multiplier, self.num_labels)
                )
            elif self.clf_gelu_head:
                self.logger.info('using 2 layer gelu classifier head')
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(self.config.hidden_size * hidden_size_multiplier, self.clf_hidden_dim),
                    torch.nn.Dropout(self.clf_drop_prob),
                    torch.nn.GELU(),
                    torch.nn.Linear(self.clf_hidden_dim, self.num_labels)
                )
            elif self.clf_roberta_head:
                self.logger.info('using 2 layer roberta classifier head')
                self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(self.clf_drop_prob),
                    torch.nn.Linear(self.config.hidden_size * hidden_size_multiplier, self.config.hidden_size * hidden_size_multiplier),
                    torch.nn.Tanh(),
                    torch.nn.Dropout(self.clf_drop_prob),
                    torch.nn.Linear(self.config.hidden_size * hidden_size_multiplier, self.num_labels)
                )
            elif self.use_bert_pooling_out:
                if self.use_cls_token or self.use_second_cls_token or self.custom_cls_tokens or self.masked_lm:
                    raise ValueError('use_bert_pooling_out is incompatible with use_cls_token, '
                                     'use_second_cls_token, custom_cls_tokens, masked_lm')
                if not isinstance(self.base_enc_model, BertModel):
                    raise ValueError('use_bert_pooling_out requires base_enc_model to be a BertModel')
                self.logger.info('using pooling out classifier head')
                self.base_enc_model.pooler = BertPooler(self.config)
                self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(self.clf_drop_prob),
                    nn.Linear(self.config.hidden_size, self.num_labels)
                )
            else:
                self.logger.info('using 2 layer leaky relu classifier head')
                self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(self.clf_drop_prob),
                    torch.nn.Linear(self.config.hidden_size*hidden_size_multiplier, self.clf_hidden_dim),
                    torch.nn.Dropout(self.clf_drop_prob),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.clf_hidden_dim, self.num_labels)
                )


    def freeze_pretrained(self):
        for param in self.base_enc_model.parameters():
            param.requires_grad = False

    def unfreeze_pretrained(self):
        for param in self.base_enc_model.parameters():
            param.requires_grad = True

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.base_enc_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outs_tensors = []

        if self.use_bert_pooling_out:
            outs_tensors.append(outputs.pooler_output)

        if self.use_cls_token:
            outs_tensors.append(outputs.last_hidden_state[:, 0])

        if self.use_second_cls_token:
            second_cls_filter = input_ids == 0 # TODO this works for roberta only
            second_cls_filter[:, 0] = False
            second_cls = outputs.last_hidden_state[second_cls_filter]
            outs_tensors.append(second_cls)

        if self.custom_cls_tokens is not None and len(self.custom_cls_tokens) > 0:
            for token in self.custom_cls_tokens:
                cls_filter = input_ids == token
                cls = outputs.last_hidden_state[cls_filter]
                outs_tensors.append(cls)

        masked_probas = None
        if self.masked_lm:
            if self.masked_lm_new_token is not None:
                masked_token_filter = input_ids == self.masked_lm_new_token
            else:
                masked_token_filter = input_ids == self.mask_token_id
            masked_repr = outputs.last_hidden_state[masked_token_filter].reshape(len(input_ids), -1)
            if self.return_masked_probas:
                masked_probas = self.lm_head(outputs[0])[masked_token_filter].reshape(len(input_ids), self.masked_lm_n_tokens, -1)
                masked_probas = torch.nn.functional.softmax(masked_probas, dim=2)
            outs_tensors.append(masked_repr)

        stance_loss = torch.tensor(0.0)
        if self.masked_lm_verbalizer:
            logits = self.lm_head(masked_repr)[:, self.verbalizer_token_ids]
        else:
            outs = torch.cat(outs_tensors, dim=1)
            logits = self.classifier(outs)
            if self.contrastive_loss or self.contrastive_loss_v2 or self.contrastive_loss_v3:
                stance_loss = self.stance_loss(outs, labels)

        loss = None
        if labels is not None:
            if self.weighted_loss:
                loss = calc_weighted_cross_entropy(labels, logits, self.num_labels, sqrt=self.weighted_loss_sqrt)
            else:
                if self.class_weights is None:
                    loss_fct = CrossEntropyLoss()
                else:
                    loss_fct = CrossEntropyLoss(weight=self.class_weights.to(self.device))
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss = ((1-self.contrastive_lambda)*loss + stance_loss * self.contrastive_lambda) \
                    if (self.contrastive_loss or self.contrastive_loss_v2 or self.contrastive_loss_v3) else loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return StanceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            masked_probas=masked_probas
        )


def get_trainer(checkpoint_dir, test_df=None, batch_size=128):
    model = StanceEncoderModel.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    test_dataset = None
    if test_df is not None:
        dev_encodings = tokenizer(list(test_df['text']), list(test_df['target']), truncation=True, padding=True)
        test_dataset = StanceDataset(dev_encodings, list(test_df['class']))

    trainer = Trainer(
        model,
        TrainingArguments('stance-infer', per_device_eval_batch_size=batch_size),
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics_f1,
        tokenizer=tokenizer,
    )

    if test_df is None:
        return trainer
    else:
        return trainer, test_dataset


def f1_macro_fa_score(labels, preds):
    clf_report = classification_report(labels, preds, output_dict=True, zero_division=0)
    f1_favor = clf_report['0']['f1-score']
    f1_against = clf_report['1']['f1-score']
    return (f1_favor + f1_against)/2


def compute_metrics_f1(p: EvalPrediction):
    preds = p.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    return {'f1_micro': f1_score(p.label_ids, preds.argmax(1), average='micro', zero_division=0),
            'f1_macro': f1_score(p.label_ids, preds.argmax(1), average='macro', zero_division=0),
            'f1_macro_fa': f1_macro_fa_score(p.label_ids, preds.argmax(1))}

