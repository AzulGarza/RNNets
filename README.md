# RNNets 

## Contenido

El contenido no pretende ser exhaustivo, más bien servirá de guía siguiendo lo establecido en la [planeación](Datómico-NLP.pdf).  


### TODO:
- [ ] Notas DL Book
- [ ] Notas Coursera 
- [ ] Notas Udacity RNNets
- [X] Implementación scratch
- [X] Implementación Pytorch (Python Front-end)
- [ ] Implementación Fastai
- [ ] Implementación Pytorch (C++ Front-end)
- [ ] Implementación Pytorch custom C++ and CUDA extensions
- [ ] Extending TorchScript with custom C++ Operators
- [ ] Producción de acuerdo a tutorial pytorch.


## Links importantes:

- [Recurrent Neural Networks Standford Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)
- [Deep Learning Book: Capítulo Modelos Secuenciales](https://www.deeplearningbook.org/contents/rnn.html)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Karpathy blog: The Unreasonable Effectiveness of RNNets](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Clase Karpathy Standford](https://www.youtube.com/watch?v=iX5V1WpxxkY)
- [Edwin Chen Blog: Exploring LSTMs](http://blog.echen.me/2017/05/30/exploring-lstms/)
- [GLUE Benchmark (Mencionado en el el pdf de la planeación de Datómico-NLP](https://arxiv.org/pdf/1804.07461.pdf).

## Instalación
1. `git clone https://github.com/datomicomx/RNNets.git`
2. `cd RNNets`
3. `conda env update`
4. `source activate datomico-NLP`


### 1. Implementación sólo numpy
*Aunque no es la manera más eficiente de hacer un modelo de lenguaje, la implementación con numpy desde cero está para que podamos entender de donde sale cada cosa, por favor si ven un error creen Issues para resolverlos.*

Es un modelo de lenguaje a nivel carácter utilizando la arquitectura de Recurrent Neural Networks. Para referencias ver Deep Learning Book: [Capítulo RNNets](https://www.deeplearningbook.org/contents/rnn.html)


1. Realizar paso de instalación.
2. `python rnn_scratch.py` (Por fa ahorita no corran los tests, hay unos errores de **floating points** que tengo que ver)
3. El modelo de lenguaje es para generar nombres y da resultados razonables. Aquí algunos nombres:
    1. Japonés: Yoshimara, Toko, Miyoshi.
    2. Francés: Getrin, Seran, Balle, Sabian.
    3. Alemán: Maußstanhin, Köstz, Wierhner.
    4. Chino: Lin, Yin, Tao, Shuaou-yang. 
	5. Nombres de dinosaurios: Itosaurus, Trnanatrax, Yrosaurus, Trodon, Macaesaurus.


### 2. Implementación Pytorch (Python Front-end)
[Link Pytorch](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html).

### Nota:
- El env **datomico-NLP** es el que estoy usando para todo el desarrollo de datómico exclusivo de NLP.

