import BiasDetection.BiasDetectionMetrics as BiasDetectionMetrics



# print('---------GPT2---------')
# causalObj = BiasDetectionMetrics.CausalLMBiasDetection(model_class='gpt2')

# print('---------Hellinger Distance---------')
# causalObj.hellingerDistance()
# print()
# print()
# print()
# print()
# print()
# print('---------Tok K Overlap(gender)---------')
# causalObj.topKOverlap(bias_type='gender')
# print()
# print()
# print()
# print()
# print()
# print('---------WeatScore(Gender)---------')
# causalObj.WeatScore(bias_type='gender')
# print()
# print()
# print()
# print()
# print()
# print('---------StereoScore(all)---------')
# causalObj.stereoSetScore(bias_type='all')
# print()
# print()
# print()
# print()
# print()
# print('---------Top K Percentage---------')
# causalObj.topKPercentage()
# print()
# print()
# print()
# print()
# print()
# print('---------Log Probability(Religion)---------')
# causalObj.logProbability(bias_type='religion')
# print()
# print()
# print()
# print()
# print()


# print('---------Bert---------')
# maskedObj = BiasDetectionMetrics.MaskedLMBiasDetection(model_class='bert-base-uncased')
# print('---------Log Probability---------')
# maskedObj.logProbability(bias_type='gender')
# print()
# print()
# print()
# print()
# print()
# print('---------F1 Score---------')
# maskedObj.F1Score(bias_type='gender')
# print()
# print()
# print()
# print()
# print()
# print('---------StereoScore(all)---------')
# maskedObj.stereoSetScore(bias_type='all')
# print()
# print()
# print()
# print()
# print()
# print('---------Top K Percentage---------')
# maskedObj.topKPercentage()
# print()
# print()
# print()
# print()
# print()
# print('---------WeatScore(Health)---------')
# maskedObj.WeatScore(bias_type='health')

# Load model directly
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import BertForMaskedLM

# model = AutoModel.from_pretrained("microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL")

# from transformers import BioGptTokenizer, BioGptForCausalLM
# model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
# tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = BertForMaskedLM.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


maskedObj = BiasDetectionMetrics.MaskedLMBiasDetection(use_pretrained=False, model = model, tokenizer = tokenizer)

# maskedObj.logProbability(bias_type='gender')

# print('---------F1 Score---------')
# maskedObj.F1Score(bias_type='gender')

# print('---------StereoScore(all)---------')
# maskedObj.stereoSetScore(bias_type='all')

# print('---------Top K Percentage---------')
# maskedObj.topKPercentage()

# print('---------WeatScore(Health)---------')
# maskedObj.WeatScore(bias_type='health')

# del tokenizer
# del model
# torch.cuda.empty_cache()
# with torch.no_grad():
#     del tokenizer
#     del model

