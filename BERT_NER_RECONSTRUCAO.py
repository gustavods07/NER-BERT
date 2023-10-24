# install pytorch: check https://pytorch.org/
# !pip install transformers 
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# parameters
model_name = "pierreguillou/ner-bert-large-cased-pt-lenerbr"
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = "Havia um homem chamado Rafael da Silva Peixoto da costa, um trabalhador dedicado que vivia em uma pequena cidade no interior do Brasil. Ele trabalhava como agricultor, cultivando café em suas terras, mas sempre sonhava em expandir seus negócios e melhorar a qualidade de vida de sua família. Ele sabia que para realizar seus sonhos, precisava de apoio financeiro e, finalmente, decidiu ir ao Banco do Brasil S/A."

# tokenization
inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt")
tokens = inputs.tokens()

# get predictions
outputs = model(**inputs).logits
predictions = torch.argmax(outputs, dim=2)


texto = []
# print predictions
for token, prediction in zip(tokens, predictions[0].numpy()):
    #print(token,model.config.id2label[prediction])
    #token_ = token.replace("##","")
    # texto ----- variavel para armazenar o label e o token associado 
    texto.append((model.config.id2label[prediction],token))



# função para diferenciar entidades diferentes de acordo com o label e com os indices dos tokens
def diferenciar(lista):
    adj = [] # vetor com os indices adjacentes
    agrupados = [] # vetor segregando grupos com indices adjacentes
    for i in range(1,len(lista)):
        if abs(lista[i] -lista[i-1]) == 1:
            #vizinhos
            
            if lista[i-1] not in adj:
                adj.append(lista[i-1])
            adj.append(lista[i])
        else:
            if adj != []:
                agrupados.append(adj)
            adj = []
    agrupados.append(adj)
           

    return agrupados

def retriever(texto):
    indexes_pessoas = [] # vetor para armazenar os indices de todos os tokens com o label PESSOA
    indexes_orgs = [] # vetor para armazenar os indices de todos os tokens com o label ORGANIZACAO
    for index,token in enumerate(texto):
        # se o token for catalogado como entidade ou parte de uma entidade:
        if token[0][0] != 'O':
            entidade = token[0].split('-')[1]
            # se o token for relacionado ao label PESSOA
            if entidade == 'PESSOA':
                indexes_pessoas.append(index)
            # se o token for relacionado ao label PESSOA
            if entidade == 'ORGANIZACAO':
                indexes_orgs.append(index)
                #print(index,token[0])

    p_ids = diferenciar(indexes_pessoas) # indexes agrupados por pessoa
    o_ids = diferenciar(indexes_orgs) # indexes agrupados por organização

    pessoas = []
    orgs = []

    for pessoa_ids in p_ids:
        nome = ''
        #print(pessoa_ids)
        for index in pessoa_ids:
            string = texto[index][1]
            nome = nome + ' ' + string 
        nome = nome.replace(' ##','')
        nome = nome.replace(" ", "", 1)
        pessoas.append(nome)

    for org_ids in o_ids:
        nome = ''
        #print(org_ids)
        for index in org_ids:
            string = texto[index][1]
            nome = nome + ' ' + string 
        nome = nome.replace(' ##','')
        nome = nome.replace(" " ,"", 1)
        orgs.append(nome)






    return pessoas,orgs

pessoas,orgs = retriever(texto)

print(pessoas)
print("----------------")
print(orgs)


