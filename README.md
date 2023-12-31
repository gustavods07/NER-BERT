# NER-BERT
Modelos baseados em BERT (Bidirectional Encoder Representations for Transformers) podem ser utilizados para reconhecer entidades em texto (NER- Named Entity Recognition) tais como pessoas, organizações e datas. O modelo pierreguillou/ner-bert-large-cased-pt-lenerbr¹ foi treinado com base no dataset lener-br² para reconhecer entidades citadas em documentos jurídicos.

Entretanto, durante o processo de reconhecimento, algumas palavras não existentes no 'vocabulário' são segmentadas em tokens (sequências de caracteres que representam unidades semânticas) menores. Essa segmentação pode repartir unidades, tornando-as incompletas e redundantes. 

Por exemplo, o nome _Ântonio José da Costa e Silva_ poderia ser reconhecido como duas entidades distintas: _'Antonio Jo'_ e _'sé da Costa e Silva'_ prejudicando a acurácia prática ao tentar recuperar os nomes das pessoas citadas em um documento.

Para solucionar o problema, o script _BERT_NER_RECONSTRUCAO.py_ aplica tratamentos de texto após a identificação das entidades PESSOA e ORGANIZAÇÃO via pipeline. 

A metodologia é simples porém efetiva: 
- tokens vizinhos reconhecidos como entidades do mesmo tipo são agregados como parte da mesma entidade;
- Entidades são agrupadas em listas de acordo com o _label_ (uma lista para pessoas e outra lista para organizações).

¹https://huggingface.co/pierreguillou/ner-bert-large-cased-pt-lenerbr

²https://huggingface.co/datasets/lener_br
