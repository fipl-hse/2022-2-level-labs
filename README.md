# Лабораторные работы для 2-го курса ФПЛ (2022/2023)

В рамках предмета
["Программирование для лингвистов"](XX)
в НИУ ВШЭ - Нижний Новгород.

**Преподаватели:**

* [Демидовский Александр Владимирович](https://www.hse.ru/staff/demidovs) - лектор
* Кащихин Андрей Николаевич - преподаватель практики
* Тугарёв Артём Михайлович - преподаватель практики
* Казюлина Марина Сергеевна - академический и технический ассистент

**План лабораторных работ:**

1. [Лабораторная работа №1 (нет описания)](./README.md)
    1. Дедлайн: `XX` сентября
2. [Лабораторная работа №2 (нет описания)](./README.md)
    1. Дедлайн: `XX` октября
3. [Лабораторная работа №3 (нет описания)](./README.md)
    1. Дедлайн: `XX` ноября
4. [Лабораторная работа №4 (нет описания)](./README.md)
    1. Дедлайн: `XX` декабря

## История занятий

| Дата |Тема лекции| Материалы практики                   |
|:-----|:---| :--- |
| N/A  | N/A | N/A [Листинг кода (N/A)](./README.md) |

## Литература

### Базовый уровень

1. Mark Lutz.
   [Learning Python](https://www.amazon.com/Learning-Python-5th-Mark-Lutz/dp/1449355730).
2. Хирьянов Тимофей Фёдорович. Видеолекции.
   [Практика программирования на Python 3](https://www.youtube.com/watch?v=fgf57Sa5A-A&list=PLRDzFCPr95fLuusPXwvOPgXzBL3ZTzybY)
   .
3. Хирьянов Тимофей Фёдорович. Видеолекции.
   [Алгоритмы и структуры данных на Python 3](https://www.youtube.com/watch?v=KdZ4HF1SrFs&list=PLRDzFCPr95fK7tr47883DFUbm4GeOjjc0)
   .
4. [Официальная документация](https://docs.python.org/3/).

### Продвинутый уровень

1. Mark Lutz.
   [Programming Python: Powerful Object-Oriented Programming](https://www.amazon.com/Programming-Python-Powerful-Object-Oriented/dp/0596158106)
1. J. Burton Browning.
   [Pro Python 3: Features and Tools for Professional Development](https://www.amazon.com/Pro-Python-Features-Professional-Development/dp/1484243846)
   .

## Порядок сдачи и оценивания лабораторной работы

Порядок сдачи:

1. лабораторная работа допускается к очной сдаче.
2. студент объяснил работу программы и показал её в действии.
3. студент выполнил задание ментора по некоторой модификации кода.
4. студент получает оценку:
    1. соответствующую ожидаемой, если все шаги выше выполнены и ментор удовлетворён ответом студента
    2. на балл выше ожидаемой, если все шаги выше выполнены и ментор решает поощрить студента за отличный ответ
    3. на балл ниже ожидаемой, если лабораторная работа сдаётся на неделю позже срока сдачи и выполнены критерии в 4.1
    4. на два балла ниже ожидаемой, если лабораторная работа сдаётся на две недели и позже от срока сдачи и выполнены
       критерии в 4.1

> Замечание: студент может улучшить оценку по лабораторной работе, если после основной сдачи выполнит
> задания следующего уровня сложности
> относительно того уровня, на котором выполнялась реализация.

Лабораторная работа допускается к очной сдаче, если выполнены все пункты ниже:

1. представлена в виде пулл реквеста (Pull Request, PR) с правильно составленным названием по шаблону:
   `Laboratory work #<NUMBER>, <SURNAME> <NAME> - <UNIVERSITY GROUP NAME>`.
   Пример: `Laboratory work #1, Kuznetsova Valeriya - 20FPL1`.
2. имеет заполненный файл `target_score.txt` с ожидаемой оценкой. Допустимые значения: 4, 6, 8, 10.
3. имеет "зелёный" статус - автоматические проверки качества и стиля кода, соответствующие заданной ожидаемой оценке,
   удовлетворены.
4. имеет лейбл `done`, выставленный ментором. Означает, что ментор посмотрел код студента и удовлетворён качеством кода.

## Ресурсы

1. [Таблица успеваемости](https://docs.google.com/spreadsheets/d/1haOOmZQqzo9xykCbpeJ7uP1ZgO7N_Hsrhpb1apZtiDE/edit?usp=sharing)
2. [Инструкция по запуску юнит тестов](./docs/public/tests.md)

## Что делать если в родительском репозитории есть изменения и они мне нужны?

1. Создаем `upstream` таргет в репозитории:

```bash
git remote add upstream https://github.com/fipl-hse/2021-2-level-labs
```

2. Получаем данные об изменениях в удаленном репозитории:

```bash
git fetch upstream
```

3. Обновляем свой репозиторий с изменениями из удаленного репозитория:

```bash
git merge upstream/master
```
