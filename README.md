# RNNets 

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


### Contenido

El contenido no pretende ser exhaustivo, más bien servirá de guía. En cuanto a las notas, irán en el wiki o en alguna otra parte para poder meter ecuaciones. 

La intención es que al final tengamos un compendio de NLP para que los que vayan entrando se puedan integrar más rápido porque es difícil que hayan visto algo de esto en clases.

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
