# Mineração de Dados
Um chatbot que responde a perguntas sobre os partidos políticos portugueses tendo em conta o seu programa eleitoral das eleições legislativas de 2024. Extraímos e processamos informações dos manifestos dos partidos, principalmente disponíveis em formato PDF. Utilizando Python, convertemos esses PDFs em arquivos .txt e empregamos técnicas de processamento de linguagem natural (NLP) para categorizar as medidas políticas por tema. A preparação dos dados incluiu a divisão de documentos em secções coerentes usando o *spaCy*, seguida pela geração de embeddings com o modelo BGE através da API da HuggingFace. Esses embeddings foram armazenados no Pinecone, uma base de dados vetorial otimizada para buscas por similaridade. O mecanismo de recuperação utiliza a seleção de documentos top-k com sobreposição contextual e um reranker para melhorar a precisão. Para gerar respostas às consultas dos utilizadores, experimentamos vários modelos, selecionando o Mixtral-8x7B pela sua relação custo-desempenho. Avaliamos a eficácia do sistema utilizando a framework RAGAS, que mede precisão do contexto, recall, fidelidade e relevância. O frontend, desenvolvido com React, oferece uma interface intuitiva para consultas dos utilizadores e interação com o modelo de linguagem.

## Estrutura do Repositório
- **article**: Contém o artigo final do projeto;
- **backend**: Contém o código do backend do chatbot. Para correr o backend, basta correr o comando `python app.py` dentro desta pasta. Alé, disso, esta pasta contém os dados de cada chat, users, testes realizados à pipeline e os dados dos partidos políticos;
- **frontend**: Contém o código do frontend do chatbot. Para correr o frontend, é necessário ter o Node.js instalado. Para instalar as dependências, correr o comando `npm install` e para correr o frontend, correr o comando `npm start` dentro desta pasta;~
- **presentation**: Contém a apresentação final do projeto;

## Autores
- Ana Rita Poças, pg53645
- Henrique Alvelos, pg50414
- Nelson Almeida, pg52697



