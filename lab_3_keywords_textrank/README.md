# Лабораторная работа №3
  
## Дано  

1. [Текст](./assets/article.txt) про определение уровня боли у мышей по выражению их морд, вы уже встречались с ним лабораторной работе №2  ([источник](https://nplus1.ru/news/2022/09/05/mouse-pain-face))
2. [Список](./assets/stop_words.txt) стоп-слов русского языка (предлоги, союзы, местоимения и другие неполнозначные слова русского языка). Список составлен на основе списка стоп-слов библиотеки для работы с естественным языком [NLTK](https://www.nltk.org/index.html).

Код, считывающий все перечисленные выше материалы, уже написан для вас в `start.py`. 
Вы найдете эти ресурсы в следующих переменных:
* `text` - текст статьи
* `stop_words` - список стоп слов

В настоящей лабораторной работе необходимо выделить ключевые слова.

**Важно:** В рамках данной лабораторной работы **нельзя использовать сторонние модули и модуль collections.**

**Важно:** Вы *можете* использовать функции из предыдущих лабораторных работ.

## Терминология

В данной работе вам предлагается поработать с такой структурой организации данных как *граф*. Давайте коротко рассмотрим его характеристики.

Граф – сложная нелинейная структура данных, отображающая связи между разными объектами. 

![](./assets/task_description_images/graph.jpeg)
<sub><sup>[источник картинки](https://medium.com/swlh/data-structures-graphs-50a8a032db03)</sub></sup>

На картинке выше изображен граф. Круги называются **вершинами** (vertex), соединяющие их линии - **ребрами** (egde). Как видите, может быть так, что не все вершины соединены между собой. Если у двух вершин есть общее соединяющее их ребро, то такие вершины называются **инцидентными**. Ребро, соединяющее вершину с самой собой, называется *петлей*. На данном графе нет петель.

Графы бывают **ориентированные** (directed) и **неориентированные** (undirected). В ориентированных графах ребра различаются по их направлению: есть вершина, из которой ребро исходит, и вершина, к которой оно направлено. В неориентированных графах такой различия нет.  

![](./assets/task_description_images/directed_undirected_graph.jpeg)
<sub><sup>[источник картинки](https://medium.com/swlh/data-structures-graphs-50a8a032db03)</sub></sup>

Графы также делятся на **взвешенные** и **невзвешенные**. Во взвешенных графах ребра могут иметь разный вес. Так, например, на картинке ниже изображен взвешенный граф. В этом графе ребра символизируют путь из одного города в другой, и эти пути различаются по своей сложности. 

![](./assets/task_description_images/weighted_graph.jpeg)
<sub><sup>[источник картинки](https://medium.com/swlh/data-structures-graphs-50a8a032db03)</sub></sup>

В настоящей лабораторной графы будут использованы как *способ представить текст*. В таких графах вершинами являются слова, а ребрами - связи между словами. Более подробная информация представлена в шагах №4 и №7.

## Подход к выделению ключевых фраз

У большинства задач есть несколько решений. Это применимо и к выделению ключевых слов. 

В [лабораторной работе №1](../lab_1_keywords_tfidf/README.md) вы познакомились с выделением ключевых слов с помощью метрик TF-IDF и $\chi^2$. 
В [лабораторной работе №2](../lab_2_keywords_cooccurrence/README.md) был использован алгоритм RAKE.
В настоящей лабораторной работе вам предлагается реализовать алгоритм [TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf). 

Данный алгоритм основан на алгоритме [PageRank](http://papers.cumincad.org/data/works/att/2873.content.pdf), который используется для расчета веса веб-страниц. Представляется, что информацию о важности веб-страниц можно извлечь из того, как эти веб-страницы ссылаются друг на друга. Именно поэтому представление в виде графа оказывается очень удобным здесь: если представить, что ссылку одной страницы на другую можно закодировать в виде ребра между двумя страницами, то при помощи нескольких итераций не очень хитрых вычислений можно определить оптимальные веса для каждой из вершин.
В TextRank, соответственно, вместо страниц вершинами являются слова, а вместо ссылок учитывается совместная встречаемость. Больше деталей о работе алгоритма вы найдете в шагах №5 и №8.

## Что надо сделать  
### Шаг 0. Подготовка (проделать вместе с преподавателем на практике).  
  
1. Изменить файлы `main.py` и `start.py`  
2. Закоммитить изменения и создать новый pull request

Обратите внимание, что для создания нового pull request создавать новый форк не нужно! Достаточно закрыть старый pull request, в котором вы работали над первой лабораторной, и открыть новый. 
При этом чтобы все изменения родительского репозитория (в том числе связанные с новой лабораторной) появились у Вас локально, нужно сделать `git pull`.
Инструкция, как открыть pull request, лежит [здесь](https://github.com/fipl-hse/2022-2-level-labs/blob/main/docs/public/starting_guide_ru.md#%D1%81%D0%BE%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5-pull-request).
  
**Важно:** Код, выполняющий все действия от предобработки текста до выделения ключевых слов, должен быть написан в `start.py`. 
Для этого реализуйте функции в модуле `main.py` и импортируйте их в `start.py`.
  
```py  
if __name__ == '__main__':  
 # your code goes here
```
В рамках данной лабораторной работы **нельзя использовать сторонние модули и модуль collections.**  
  
  
### Шаг 1. Очистка и токенизация текста при помощи `TextPreprocessor`

В настоящей лабораторной за всю предобработку текста отвечает класс `TextPreprocessor`. Именно внутри него должен быть реализован весь пайплайн преобразования текста из монолитной неочищенной строки в корректные токены. Давайте рассмотрим, какие методы для этого необходимы.

#### Шаг 1.1 Конструктор класса

Для создания объекта `TextPreprocessor` нам необходимо знать, что именно нужно удалить из неочищенной строки. Именно поэтому в конструктор передаются следующие аргументы:

* `_stop_words` - кортеж из стоп-слов
* `_punctuation` - кортеж, содержащий в себе все подлежащие удалению символы

Внутри конструктора необходимо сохранить пришедшие на вход аргументы в одноименные атрибуты класса. Таким образом, у объекта класса `TextPreprocessor` должны быть следующие поля: `_stop_words`, `_punctuation`.


Интерфейс:   
```py  
class TextPreprocessor:
    def __init__(self, stop_words: tuple[str, ...], punctuation: tuple[str, ...]) -> None:
        pass
```  

#### Шаг 1.2 Очистка и токенизация текста

Этот метод препроцессора, как вытекает из названия, очищает текст от знаков препинания и делит его на отдельные токены. 

Метод принимает на вход неочищенный текст. Обратите внимание, что пунктуацию в этот метод передавать не нужно: информацию о том, какие знаки считать пунктуационными, этот метод берет из ранее сохраненной в поле экземпляра строки. 

Возвращаемым значением метода является кортеж выделенных токенов.

Интерфейс:   
```py  
_clean_and_tokenize(text: str) -> tuple[str, ...]:
    pass
```  

> Проверьте себя: почему название метода начинается с нижнего подчеркивания?

#### Шаг 1.3 Удаление стоп-слов

В этом методе происходит фильтрация токенов: стоп-слова необходимо убрать. Как и на предыдущем шаге, не нужно дополнительно сообщать методу, какие слова считаются стоп-словами, эта информация уже сохранена в самом экземпляре, нужно обращаться к нему. 

Входным аргументом метода является кортеж из токенов. Возвращает метод также кортеж из токенов, не содержащий стоп-слов.

Интерфейс:   
```py  
def _remove_stop_words(self, tokens: tuple[str, ...]) -> tuple[str, ...]:
    pass
```  

#### Шаг 1.4 Полная предобработка текста

Внутри данного метода должен быть реализован весь пайплайн предобработки текста. Функция принимает монолитную неочищенную строку и возвращает очищенные отфильтрованные токены. При реализации этого метода **необходимо** обращаться к уже определенным методам класса.

Интерфейс:   
```py  
def preprocess_text(self, text: str) -> tuple[str, ...]:
    pass
```  

### Шаг 2. Кодирование текста 

При работе с естественным языком часто прибегают к кодированию текста. Это связано в том числе с тем, что многие операции дешевле проводить над числами, нежели чем над последовательностью символов. Например, сравнение двух чисел происходит намного быстрее, чем сравнение двух строк.

В настоящей лабораторной мы также перейдем от строк к числам. Поможет нам в этом класс `TextEncoder`. Рассмотрим его методы.

#### Шаг 2.1 Конструктор класса

Чтобы экземпляр кодировщика мог успешно переводить токены в числа и обратно, необходимо завести внутри объекта два словаря. Один из них хранит соответствия вида `строка -> число` и называется `_word2id`, а другой хранит соответствия `число -> строка` и называется `_id2word`. Мы пока не знаем, с какими именно строками нам придется работать, поэтому нужно инициализировать эти два атрибута пустыми словарями. Позже они будут заполнены.

Конструктор ничего не принимает на вход.

Интерфейс:   
```py  
class TextEncoder:
    def __init__(self) -> None:
        pass
```  

#### Шаг 2.2 Заполнение атрибутов класса

Данный метод кодировщика принимает на вход кортеж из строк и соответствующим образом заполняет атрибуты `_word2id` и `_id2word`. Метод ничего не возвращает.
Каждой строке должно быть присвоено целое число. Одной строке присваивается ровно одно уникальное число. **Минимальное число среди присвоенных чисел должно быть не меньше 1000.**

Рассмотрим пример. Пусть дана следующая последовательность токенов: `('в', 'лесу', 'родилась', 'ёлочка', 'в', 'лесу', 'она', 'росла')`
В этой последовательности всего 6 разных строк: `'в'`, `'лесу'`, `'она'`, `'родилась'`, `'росла'`, `'ёлочка'`. Присвоим каждой из этих строк по одному уникальному числу больше 1000. Допустим, соответствие получается такое:

* `'в'` - `1009`
* `'лесу'` - `1001`
* `'она'` - `1005`
* `'родилась'` - `1101`
* `'росла'` - `1061`
* `'ёлочка'` - `1041`

Тогда атрибут `_word2id` должен выглядеть следующим образом: `{'в': '1009', 'л': '1001', 'о': '1005', 'р': '1061', 'ё': '1041'}`. В `_id2word`, соответственно, ключи и значения меняются местами: `{'1009': 'в', '1001': 'л', '1005': 'о', '1101': 'р', '1061': 'р', '1041': 'ё'}`.

Если вы все сделали правильно, то длина данных атрибутов должна совпадать.

Интерфейс:   
```py  
def _learn_indices(self, tokens: tuple[str, ...]) -> None:
    pass
```  
   
#### Шаг 2.3 Преобразование последовательности строк в последовательность чисел
   
Этот метод принимает на вход кортеж из строк и возвращает кортеж из чисел.
Внутри функции вызывается метод `_learn_indices`, который задает соответствие между строками и числами.
Выбор числа, на которое необходимо заменить каждую из строк, определяется атрибутом `_word2id`.

Например, на вход пришла такая последовательность: `'она', 'родилась', 'в', 'лесу'`. Обращаясь к атрибутам, сформированным на предыдущем шаге, мы должны получить следующую закодированную последовательность: `1005, 1101, 1009, 1001`.

В случае, если на вход приходит пустая последовательность токенов,  возвращает `None`. 

Интерфейс:   
```py  
def encode(self, tokens: tuple[str, ...]) -> Optional[tuple[int, ...]]:
    pass
```  

#### Шаг 2.4 Декодирование токенов

Превращение слов в числа упрощает работу с ними, но никак не их читаемость. Мы стремимся выделить из текста ключевые слова, а не числа, поэтому нужно реализовать обратное преобразование. Метод принимает на вход кортеж из чисел и возвращает кортеж из строк. Информацию о соответствии между числами и строками необходимо брать из атрибутов объекта.

Например, на вход пришла такая последовательность: `1061, 1009, 1001, 1041`. Обращаясь к атрибутам, сформированным на предыдущем шаге, мы должны получить следующую закодированную последовательность: `'росла', 'в', 'лесу', 'ёлочка'`.

В случае, если на вход приходит последовательность, в которой есть слова, не содержащиеся в атрибутах экземпляра класса, возвращается `None`.

Интерфейс:   
```py  
def decode(self, encoded_tokens: tuple[int, ...]) -> Optional[tuple[str, ...]]:
    pass
```  

### Шаг 3. Выделение пар слов. Выполнение Шагов 1-3 соответствует 4 баллам

Используемый в этой лабораторной работе алгоритм опирается на совместную встречаемость слов. В этом шаге необходимо научиться ее извлекать. 

Функция принимает на вход список закодированных токенов и ширину окна. Возвращаемым значением является кортеж пар токенов, которые встречаются рядом. Мы считаем, что слова встречаются рядом, если они помещаются в одно *окно*. Окно - это отрезок списка токенов определенной длины. Давайте рассмотрим пример. 

* последовательность токенов: `(1, 2, 3, 4, 3)`
* длина окна: `3`
* выделенные пары слов: `((1, 2), (1, 3), (2, 3), (3, 4), (2, 4))`

Обратите внимание, что пара `(3, 3)` не была выделена: нас интересуют *разные* токены, умещающиеся в одно окно.

Если на вход приходит пустая последовательность токенов, нецелочисленное значение длины окна или длина окна меньше 2, то возвращается `None`. 

Интерфейс:   
```py
def extract_pairs(tokens: tuple[int, ...], window_length: int):
    pass
```  

Продемонстрируйте работу данной функции в файле `start.py`, передав ей закодированный токенизированный текст. 

### Шаг 4. Представление текста в виде графа. Матрица смежности

Мы хотим задать такой граф, в котором вершинами будут являться токены, а ребрами - связи между токенами. Мы строим *невзвешенный* граф: мы не дифференцируем связи между словами по их силе/близости/др. Если токены встречаются на расстоянии не больше длины окна, то между ними есть ребро, то есть эти вершины *инцидентны*. Если слова никогда не образовывали такую пару, то они не инцидентны. 

Кроме этого, предлагается также не учитывать порядок слов и сделать граф *неориентированным*: главное, что слова встретились вместе в каком-то окне, а порядок, в котором они там появились, нас не интересует

> В оригинальной статье по алгоритму PageRank вы увидите, что авторы используют ориентированный граф. В более поздних работах, описывающих применение этого алгоритма для взвешивания слов в тексте, было отмечено, что алгоритм можно также применять и к неориентированным графам, в этом случае процесс обучения может идти более плавно, но он неизбежно приведет к тем же результатам.  ([ссылка](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf))

Существует несколько способов задать граф. Один из распространенных - матрица смежности. Давайте познакомимся с этим форматом хранения графа. 

Допустим, у нас вот такой неориентированный невзвешенный граф:

![](./assets/task_description_images/custom_graph.jpg)

Матрица смежности представляет собой не что иное, как таблицу. В нашем случае незаполненная матрица смежности выглядит вот так.

|   |A   |B   |C   |D   |
|---|---|---|---|---|
|A   |   |   |   |   |
|B   |   |   |   |   |
|C   |   |   |   |   |
|D   |   |   |   |   |

В ней одинаковое количество столбцов и строк. При этом каждая строка и каждый столбец соответствуют одной какой-либо вершине графа. Например, на пересечении строки, соответствующей вершине А, и столбца, соответствующего вершине В, будет записано число, отражающее наличие или отсутствие ребра между этими вершинами. Давайте посмотрим, как она выглядит в заполненном виде.

|   |A  |B  |C  |D  |
|---|---|---|---|---|
|A   |0   |1  |1   |0  |
|B   |1   |0  |0   |1  |
|C   |1   |0  |0   |0  |
|D   |0   |1  |0   |1  |

> Проверьте себя: почему у нас получилась симметричная матрица? Может ли матрица смежности быть несимметричной? Каким тогда должен быть граф?

Ниже описаны шаги реализации матрицы смежности.

#### Шаг 4.1 Конструктор

Инициализация матрицы смежности не требует никаких входных аргументов. Конструктор задает необходимые атрибуты класса.

У каждого объекта графа **обязательно** должны быть следующие атрибуты:
* `self._matrix` - таблица, кодирующая связи между вершинами. Ожидаемый тип данных, хранящийся в этом атрибуте: `list[list[int]]]`. Пока в граф не добавлено ни одной вершины, можно задать это поле пустым списком.

```py
class AdjacencyMatrixGraph:
    def __init__(self) -> None:
        pass
```

#### Шаг 4.2 Добавление ребра

В данной лабораторной граф является изменяемой структурой данных: мы можем добавлять вершины и ребра после инициализации объекта.

Ответственным за это является метод `add_edge`. Он принимает на вход две вершины. Внутри функции записывается наличие ребра между этими вершинами в поле `_matrix`. В случае, если вершины, пришедшей в качестве аргумента, в графе еще нет, необходимо ее добавить. 

Петли, то есть ребра, соединяющие вершину с самой собой, добавлять нельзя. 

Если добавление ребра прошло успешно, то метод возвращает 0. Если в качестве обоих входных аргументов выступила одна и та же вершина, то возвращается -1.

Интерфейс:
```py
add_edge(self, vertex1: int, vertex2: int) -> int:
    pass
```

#### Шаг 4.3 Получение веса ребра

Для использования графа необходимо иметь возможность запросить у него информацию о том, связаны ли две вершины ребром. Для этого нужно определить метод `is_incidental`. Метод принимает на вход два аргумента: вершина №1 и вершина №2. Метод возвращает 1, если эти вершины инцидентны, в противном случае возвращается 0.

Если какая-либо из запрошенных вершин отсутствует в графе, возвращается -1.

Интерфейс:
```py
def is_incidental(self, vertex1: int, vertex2: int) -> int:
    pass 
```

#### Шаг 4.4 Получение информации обо всех вершинах

Также необходимо реализовать возможность узнать, какие вершины есть в графе. Для этого следует реализовать метод `get_vertices`. Он не принимает никаких аргументов. Возвращаемым значением является кортеж вершин сохраненного графа.

Интерфейс:
```py
def get_vertices(self) -> tuple[int, ...]:
    pass
```

#### Шаг 4.5 Подсчет количества инцидентных вершин

Для работы алгоритма TextRank очень важно знать, с каким количеством вершин каждая конкретная вершина имеет общие ребра. Это называется `InOut score`. 

> В оригинальной статье Вы можете увидеть, что вместо `InOut score` различаются `In score` и `Out score`. Такое разделение обусловлено ориентированностью графа. В данной лабораторной мы реализуем неориентированный граф, поэтому для нас имеет смысл только количество инцидентных вершин.

Функция принимает на вход вершину и возвращает количество инцидентных с ней вершин. Если на вход приходит вершина, которой нет в графе, возвращается -1.

Интерфейс:
```py
def calculate_inout_score(self, vertex: int) -> int:
    pass
```

#### Шаг 4.6. Заполнение экземпляра графа 

Для упрощения работы с экземпляром графа реализуем метод, который заполнит его всеми вершинами и ребрами, которые можно извлечь из последовательности закодированных токенов.

Метод принимает на вход последовательность токенов и длину окна. Внутри функции происходит выделение пар слов и добавление соответствующих ребер в граф. Необходимо обращаться к функции `extract_pairs` и к методу `add_edge`. Метод ничего не возвращает.

Интерфейс:
```py
def fill_from_tokens(self, tokens: tuple[int, ...], window_length: int) -> None:
    pass
```


### Шаг 5. Алгоритм TextRank. Классическая реализация

В алгоритме TextRank мы пытаемся оценить важность токенов путем анализа их совстречаемости. При этом оценка важности происходит итеративно. 

1. Изначально все веса инициализируются единицей. Можно выбрать любое число, но традиционно это `1`. 
2. После этого вес каждой вершины пересчитывается по этой формуле:
$$S(V_{i}^{k}) = (1 - d) + d \cdot \sum_{j \in InOut(V_{i})} \frac{1}{|InOut(V_{j})| } S(V_{j}^{k-1})$$

    Обозначения:
    * $V$ - вершина графа (то есть слово)
    * $d$ - damping factor, то есть вероятность перейти от этой вершины к любой другой, его значение обычно равно 0.85
    * $InOut(V)$ - вершины, инцидентные вершине $V$
    * $InOut(V)$ - количество вершин, инцидентных с $V$ (то есть, как мы назвали это в предыдущем пункте, `InOut score`)
    * $k$ - это номер итерации: при переподсчете веса вершины мы опираемся на веса инцидентных с ней вершин с *прошлых итераций*
    
    Таким образом, чтобы посчитать новый вес для новой вершины, нужно:
    * перебрать все вершины, инцидентные с данной вершиной
    * для каждой из таких вершин найти ее `InOut score`, использовать его как знаменатель в дроби $\frac{1}{|InOut(V_{j})|}$ и умножить полученное значение на вес этой же вершины
    * полученные значения просуммировать и умножить на $d$
    * далее результат умножения сложить с $(1 - d)$, и результат такого сложения становится новым весом рассматриваемой вершины

3. После такой оценки важности необходимо узнать, насколько сильно веса вершин изменились: для этого из веса каждой вершины вычитается ее предыдущий вес. Полученные разности суммируются и сравниваются с *порогом сходимости* - это какое-то маленькое заранее заданное число. В нашем случае оно равно 0.0001. Если сумма разностей весов оказалась меньше, чем это маленькое число, то можно сказать, что вес вершин почти не изменился. Это значит, что оптимальные веса были найдены, алгоритм завершается. В противном случае мы продолжаем переоценку важности вершин, возвращаясь в пункт 2.

#### Шаг 5.1 Конструктор класса и обучение

При инициализации класса на вход приходит один аргумент: заполненный объект графа. Внутри конструктора задаются необходимые для работы атрибуты. Рассмотрим их внимательнее:
* `_graph` - здесь хранится пришедший на вход заполненный граф, который, как мы помним, является репрезентацией текста
* `_damping_factor` - как упоминалось ранее, это константное значение, нужное для того, чтобы алгоритм не зациклился и отражающее вероятность перейти от конкретной вершины к любой другой; установите значение данного атрибута как 0.85
* `_convergence_threshold` - это константное значение, отражающее максимально допустимую разницу между весами до и после обновления: если после очередной итерации пересчета весов вершин сумма разностей вершин не превышает этот порог, то мы считаем, что оптимальные веса найдены; установите значение данного атрибута как 0.0001
* `_max_iter` - это константное значение, отражающее максимально допустимое количество итераций обновления весов: теоретически на вход может прийти очень сложный граф, в котором невозможно подобрать стабильные веса, и чтобы не попасть в бесконечный цикл, мы ограничиваем максимальное количество итераций; установите значение данного атрибута как 50
* `_scores` - веса вершин, здесь ключами являются вершины, а значениями их вес, отражающий их важность; инициализируйте этот атрибут пустым словарем

Интерфейс:
```py
def __init__(self, graph: Union[AdjacencyMatrixGraph, EdgeListGraph]) -> None:
    pass
```

#### Шаг 5.2 Обновление веса конкретной вершины

Реализуйте обновление веса конкретной вершины. Метод принимает на вход вершину, вес которой необходимо пересчитать, список инцидентных с ней вершин и словарь с весами вершин после предыдущей итерации обновления. 

Новый вес вершины необходимо рассчитать по формуле из Шага №6. Для получения информации о значении $d$ необходимо обратиться к соответствующему атрибуту класса, информацию о количестве инцидентных вершин нужно взять при помощи метода графа `calculate_inout_score`. Доступ к графу также происходит через атрибуты класса.

Метод ничего не возвращает. Вместо этого он перезаписывает вес вершины в словаре, хранящемся в атрибуте `scores`. 

Интерфейс: 
```py
def update_vertex_score(self, vertex: int, incidental_vertices: list[int], scores: dict[int, float]) -> None:
    pass
```

#### Шаг 5.3 Итеративное обновление весов вершин

Метод итеративного переподсчета важности токенов написан для вас: `score_vertices`. Метод ничего не принимает на вход и ничего не возвращает. Вместо этого он изменяет атрибут `_scores`.

Внимательно изучите код и приготовьтесь объяснить его на защите.

#### Шаг 5.4 Получение информации о весе вершин

Обращаться к защищенному атрибуту экземпляра извне - плохая практика. Поэтому необходимо реализовать метод, возвращающий словарь с весами вершин. 

Метод ничего не принимает на вход. Метод возвращает словарь, ключами в котором являются вершины, а значениями - их вес.

Интерфейс:
```py
def get_scores(self) -> dict[int, float]:
    pass
```

#### Шаг 5.5 Получение ключевых слов

Наконец, пора определять метод, возвращающий ключевые слова. 

Метод принимает на вход аргумент, обозначающий требуемое количество ключевых слов. Метод возвращает последовательность слов с самым высоким весом. 

```py
def get_top_keywords(self, n_keywords: int) -> tuple[int, ...]:
    pass
```

### Шаг 6. Демонстрация выделения ключевых слов. Выполнение Шагов 1-6 соответствует 6 баллам

Продемонстрируйте выделение ключевых слов в файле `start.py`. Не забудьте предобработать текст, закодировать его, создать экземпляр графа, заполнить его, итеративно вычислить вес его вершин, отобрать 10 самых важных токенов и декодировать их. 

### Шаг 7. Представление текста в виде графа. Список ребер

Матрица смежности не единственный способ реализовать граф. В этом шаге мы познакомимся с таким представлением как список ребер. Обычно под этим понимается буквально список ребер. В настоящей лабораторной это больше похоже на список инцидентных вершин. Давайте разберемся, что имеется в виду. 

Вспомним наш граф.

![](./assets/task_description_images/custom_graph.jpg)

Предлагается хранить информацию о ребрах графа при помощи словаря, в котором ключами являются вершины, а значениями - список инцидентных вершин. Таким образом, для нашего графа это будет выглядеть следующим образом: 

```py
{
    A: [C, B],
    B: [A, C, D],
    C: [A, B],
    D: [B]
}
```

Поэтапно рассмотрим реализацию. 

#### Шаг 7.1 Конструктор класса 

Инициализация графа не требует никаких входных аргументов. Конструктор задает необходимые атрибуты класса.

У каждого объекта графа обязательно должен быть атрибут `_edges`. В нем хранится словарь, кодирующий инцидентность вершин. Ожидаемый тип данных, хранящийся в этом атрибуте, выглядит следующим образом: `dict[int, list[int]]`. Пока в граф не добавлено ни одной вершины, можно задать это поле пустым словарем.

Интерфейс:
```py
class EdgeListGraph:
    def __init__(self) -> None:
        pass
```

#### Шаг 7.2 Унификация интерфейса

Экземпляр `EdgeListGraph` должен иметь такой же интерфейс, как и `AdjacencyMatrixGraph`. Реализуйте в классе `EdgeListGraph` следующие методы:
* `add_edge`
* `is_incidental`
* `get_vertices`
* `calculate_inout_score`
* `fill_from_tokens`

Интерфейс методов и возвращаемое значение в каждом из случаев остаются такими же, как описано для `AdjacencyMatrixGraph`.

#### Шаг 7.3 Демонстрация выделения ключевых слов на базе `EdgeListGraph`

Интерфейс классов `AdjacencyMatrixGraph` и `EdgeListGraph` совпадает неслучайно: представляется, что алгоритм выделения слов не должен зависеть от того, как именно организуется информация внутри графа. 

Выделите 10 ключевых слов при помощи алгоритма `VanillaTextRank`, передав ему заполненный объект `EdgeListGraph` в `start.py`.

### Шаг 8. Расширение классов `AdjacencyMatrixGraph` и `EdgeListGraph`. Учет позиции токенов в тексте

Попробуем улучшить алгоритм TextRank путем принятия во внимании позиций слов в тексте. Для этого необходимо добавить соответствующий функционал в абстракции графов. 

#### Шаг 8.1 Добавление атрибутов в конструкторы `AdjacencyMatrixGraph` и `EdgeListGraph`

Добавьте в конструкторы классов следующие атрибуты:
* `_positions`: данный атрибут должен хранить в себе информацию обо всех позициях каждой из вершин графа, предполагаемый тип: `dict[int, list[int]]`, инициализируйте как пустой словарь
* `_position_weights`: данный атрибут должен хранить в себе информацию обо всех позиционных *весах* каждой из вершин графа, предполагаемый тип: `dict[int, float]`, инициализируйте как пустой словарь

#### Шаг 8.2 Сохранение информации о позициях токенов

В классах `AdjacencyMatrixGraph` и `EdgeListGraph` реализуйте метод, принимающий на вход последовательность закодированных токенов и заполняющий атрибут `_positions`. Значением словаря `_positions` должен быть словарь всех позиций токена в последовательности токенов. Рассмотрим пример. 

Допустим, последовательность токенов выглядит так: `(1, 256, 4, 95, 1, 420, 5)`
Токен `1` встречается на первой позиции и на пятой. Поэтому значение `positions` по ключу `1` должно быть равно `[1, 5]`.

> Обратите внимание, что в данном контексте понятие *позиция* отличается от понятия *индекс*: индексация начинается с нуля, а нумерация позиций - с единицы.

Интерфейс:
```py
def fill_positions(self, tokens: tuple[int, ...]) -> None:
    pass
```

#### Шаг 8.3 Подсчет веса вершины исходя из позиции слова в тексте

Позиционный вес вершины рассчитывается по следующей формуле:

$$p(V) = \frac{1}{position_{1}} + \frac{1}{position_{2}} + \frac{1}{position_{3}} + ...$$

$position_{i}$ обозначает индекс i-ого вхождения слова, при этом нумерация начинается с 1. Давайте посмотрим на пример. 

Пусть дана такая последовательность токенов: `(1, 2, 3, 4, 1, 7, 10)`.
Рассчитаем позиционный вес токена `1`. Токен `1` встретился два раза: первый раз в позиции 1, второй раз - в позиции 5. Поэтому его позиционный вес вычисляется так: $p(1) = \frac{1}{1} + \frac{1}{5}$. Несложно посчитать, что $p(1) = 1.2$. 

Если токен встречается в тексте 100 раз, то в формуле его позиционного веса будет 100 слагаемых, если один раз, то одно. Кажется, довольно просто.

Однако если мы оставим веса в таком виде, то они по своему масштабу "перекричат" значение $d$: оно, как мы помним, традиционно равно 0.85. Поэтому веса необходимо нормировать:

$$\tilde{p_{i}} = \frac{p_{i}}{p_{1} + p_{2} + ... + p_{|V|}}$$

Здесь $\tilde{p_{i}}$ - это нормированный позиционный вес токена $i$,   $p_{i}$ - ненормированный позиционный вес токена $i$ (то есть обычный, который мы посчитали выше), а $p_{1} + p_{2} + ... + p_{|V|}$ - это сумма ненормированных весов всех вершин. Таким образом, нормированные веса не превышают 1. 

В методе `calculate_position_weights` необходимо реализовать подсчет нормированных позиционных весов для каждого из вершин графа. Метод ничего не принимает на вход: информация о вершинах графа и об их позициях в тексте извлекается через обращение к атрибутам. Метод ничего не возвращает, но изменяет словарь, лежащий в атрибуте `_position_weights`, в котором ключами являются вершины графа, а значениями - их нормированные позиционные веса. 

Интерфейс:
```py
def calculate_position_weights(self) -> None:
    pass
```

#### Шаг 8.4 Получение информации о позиционных весах

Необходимо реализовать публичный метод, возвращающий информацию о позиционных весах токенов. Метод не принимает дополнительные аргументы. Метод возвращает словарь, в котором ключами являются вершины, а значениями - их нормированные позиционные веса.

Интерфейс:
```py
def get_position_weights(self) -> dict[int, float]:
    pass
```


### Шаг 9. Алгоритм `PositionBiasedTextRank`. Учет позиции слов в тексте. Выполнение шагов 1-9 соответствует 8 баллам

У алгоритма TextRank есть множество улучшенный версий. Одной из них является модификация `PositionBiasedTextRank`. Как это понятно из названия, в данной версии в расчет весов вершин (слов) принимается позиция этого слова в тексте.

Представляется, что чем ближе к началу встречается слово и чем чаще оно встречается, тем более важным оно является. Поэтому в данной модификации формула обновления веса вершины выглядит так: 

$$S(V_{i}) = (1 - d) \cdot \tilde{p}  + d \cdot \sum_{j \in InOut(V_{i})} \frac{1}{|InOut(V_{j})| } S(V_{j})$$

Формула практически не отличается от той, которую мы использовали в шаге 5: единственным отличием является дополнительный множитель в первом слагаемом: это позиционный вес слова. Чем ближе слово к началу и чем чаще слово встречается в тексте, тем позиционный вес больше. Далее мы рассмотрим вычисление позиционных весов более подробно.

#### Шаг 9.1 Конструктор класса `PositionBiasedTextRank`

Для того чтобы иметь возможность принимать во внимание позицию слова в тексте, нам необходимо задать соответствующие атрибуты. 

Конструктор принимает на вход заполненный объект графа.

Данный класс наследуется от `VanillaTextRank`. Именно поэтому определение части атрибутов необходимо делегировать родительскому классу, обратившись к `super()`.

Кроме атрибутов, которые задаются в конструкторе родительского класса, необходимо задать поле `_position_weights` - это словарь, в котором ключами являются вершины, а значениями - их позиционный вес. Для инициализации этого атрибута обратитесь к соответствующему *методу* пришедшего на вход экземпляра графа. 

Интерфейс:
```py
class PositionBiasedTextRank(VanillaTextRank):
    def __init__(self, graph: Union[AdjacencyMatrixGraph, EdgeListGraph]) -> None:
        pass
```

#### Шаг 9.2 Обновление веса вершины

Так как в данной модификации подсчет веса вершины происходит по немного другой формуле, нежели чем в классической вариации TextRank, необходимо переопределить метод обновления веса вершины. 

Метод все также принимает на вход вершину, вес которой необходимо пересчитать, список инцидентных с ней вершин и словарь с весами вершин после предыдущей итерации обновления.

Новый вес вершины необходимо рассчитать по формуле из пункта 8. Для получения информации о значении $d$ необходимо обратиться к соответствующему атрибуту класса, информацию о количестве инцидентных вершин нужно взять при помощи метода графа `calculate_inout_score`. Доступ к графу также происходит через атрибуты класса. Наконец, информацию о позиционных весах также нужно получать через соответствующий атрибут объекта. 

Метод ничего не возвращает. Вместо этого он перезаписывает вес вершины в словаре, хранящемся в атрибуте `scores`.

Интерфейс:
```py
def update_vertex_score(self, vertex: int, incidental_vertices: list[int], scores: dict[int, float]) -> None:
    pass
```

#### Шаг 9.3 Продемонстрируйте выделение ключевых слов `PositionTextRank`

Покажите 10 самых значимых ключевых слов, выделенных при помощи класса `PositionTextRank`, в `start.py`. Для этого используйте функцию `get_top_n`.

Проделайте это 2 раза, передав при инициализации алгоритму сначала заполненный экземпляр `AdjacencyMatrixGraph`, а затем `EdgeListGraph`.

> С каким вариантом графа поиск проходит быстрее? Почему?

> Отличается ли выбор ключевых слов у разных модификаций TextRank?