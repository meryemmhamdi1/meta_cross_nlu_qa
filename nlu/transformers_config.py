from transformers import BertTokenizer, BertModel, \
    OpenAIGPTTokenizer, OpenAIGPTModel, GPT2Tokenizer, GPT2Model, \
    CTRLTokenizer, CTRLModel, TransfoXLTokenizer, TransfoXLModel, \
    XLNetTokenizer, XLNetModel, XLMTokenizer, XLMModel, \
    RobertaTokenizer, RobertaModel, XLMRobertaTokenizer, XLMRobertaModel, \
    DistilBertTokenizer, DistilBertModel, AlbertTokenizer, AlbertModel

MODELS_dict = {"BertLarge": ('bert-large-uncased', BertTokenizer, BertModel),
               "BertBaseCased": ('bert-base-cased', BertTokenizer, BertModel),
               "BertBaseMultilingualCased": ('bert-base-multilingual-cased', BertTokenizer, BertModel),
               "Xlnet_base": ('xlnet-base-cased', XLNetTokenizer, XLNetModel),
               "Xlnet_large": ('xlnet-large-cased', XLNetTokenizer, XLNetModel),
               "XLM": ('xlm-mlm-enfr-1024', XLMTokenizer, XLMModel),
               "DistilBert_base": ('distilbert-base-uncased', DistilBertTokenizer, DistilBertModel),
               "DistilBert_large": ('distilbert-large-cased', DistilBertTokenizer, DistilBertModel),
               "Roberta_base": ('roberta-base', RobertaTokenizer, RobertaModel),
               "Roberta_large": ('roberta-large', RobertaTokenizer, RobertaModel),
               "XLMRoberta_base": ('xlm-roberta-base', XLMRobertaTokenizer, XLMRobertaModel),
               "XLMRoberta_large": ('xlm-roberta-large', XLMRobertaTokenizer, XLMRobertaModel),
               "ALBERT-base-v1": ('albert-base-v1', AlbertTokenizer, AlbertModel),
               "ALBERT-large-v1": ('albert-large-v1', AlbertTokenizer, AlbertModel),
               "ALBERT-xlarge-v1": ('albert-xlarge-v1', AlbertTokenizer, AlbertModel),
               "ALBERT-xxlarge-v1": ('albert-xxlarge-v1', AlbertTokenizer, AlbertModel),
               "ALBERT-base-v2": ( 'albert-base-v2', AlbertTokenizer, AlbertModel),
               "ALBERT-large-v2": ('albert-large-v2', AlbertTokenizer, AlbertModel),
               "ALBERT-xlarge-v2": ('albert-xlarge-v2', AlbertTokenizer, AlbertModel),
               "ALBERT-xxlarge-v2": ('albert-xxlarge-v2', AlbertTokenizer, AlbertModel),
               }
