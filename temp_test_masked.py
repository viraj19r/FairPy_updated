# from audioop import bias
import BiasDetection.BiasDetectionMetrics as BiasDetectionMetrics
# import BiasMitigation.BiasMitigationMethods as BiasMitigationMethods
import sys

#Causal - gpt2, openai-gpt, ctrl, xlnet-base-cased, transfo-xl-wt103, xlm-mlm-en-2048, roberta-base
#Masked - bert-base-uncased, distilbert-base-uncased, roberta-base, albert-base-v1

#CausalMitObj = BiasMitigationMethods.CausalLMBiasMitigation(model_class='gpt2')
# MaskedMitObj = BiasMitigationMethods.CausalLMBiasMitigation(model_class='gpt2')
# model, tokenizer = MaskedMitObj.NullSpaceProjection('gpt2', 'GPT2LMHeadModel', 'gender', train_data='yelp_sm')
# model, tokenizer = MaskedMitObj.DropOutDebias('gpt2', 'gender', 'yelp_sm', 100)
#print(model)

#exit()

maskedObj = BiasDetectionMetrics.MaskedLMBiasDetection(model_class = 'bert-base-uncased')
maskedObj.topKPercentage()
# maskedObj.logProbability(bias_type='gender')
# maskedObj.F1Score(bias_type='gender')
# maskedObj.WeatScore(bias_type='health')
# maskedObj.stereoSetScore(bias_type='all')



