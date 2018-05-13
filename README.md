# recutils
LSML2018 recommender utils

## Общее описание утилиты
Разработана утилита на языке программирования Java для обучения линейных моделей (линейная модель без парных взаимодействий, FM).

Поддерживаются квадратичная функция потерь для задач регрессии и логистическая функция потерь для задач бинарной классификации. Модель без парных взаимодействий обучается с помощью SGD, для FM реализовано обучение с помощью SGD и ALS. 

Для SGD поддеживается параллельное обучение (Hogwild!). 

Поддерживается holdout-валидация (10% от обучающей выборки), при обучении выводится средняя ошибка на трейне (а если используется holdout, то и на валидации) после каждой эпохи обучения. Эпоха = один пробег по обучающей выборке для SGD, обновление всех весов модели для ALS.

Полный список опций: https://github.com/SenderOK/recutils/blob/master/src/main/java/ru.recutils/cli/CommandLineArguments.java

Обучение реализовано на основе следующих статей:

1) https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle_et_al2011-Context_Aware.pdf - правда, здесь в коде основной процедуры LearnALS допущено 2 опечатки: на строке 16 забыта звёздочка в присваиваемой переменной, на строке 23 последнее домножение должно быть не на x, а на h.

2) https://www.csie.ntu.edu.tw/~b97053/paper/Factorization%20Machines%20with%20libFM.pdf

3) https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf - Hogwild!

Утилита работает с файлами в [формате vowpal wabbit](https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format), в настоящее время без полей *tag*, *base* и неймспейсов. 

Для управления размером модели применяется hashing trick (опция *-hb/--hashing-bits*), для хеширования используется заданное число последних битов MurmurHash3. Модели сохраняются с помощью стандатного метода сериализации в Java (интерфейс Serializable).

Для тестирования некоторых процедур реализованы юнит-тесты (JUnit).

### SGD
Для SGD реализовано онлайн-обучение, в начале каждой эпохи файл с обучающей выборкой открывается заново, объекты вычитываются из файла по одному и происходит обновление весов. Это может быть сделано в несколько потоков, что даёт значительное ускорение при обучении. 

Технически это реализовано с помощью параллельных потоков Java (интерфейс Stream), которые можно использовать для ввода (стандартный пакет java.nio.file.Files), атомарных счётчиков и ConcurrentHashMap для хранения весов.

Использован sgd c постоянным шагом.

### ALS
В отличие от SGD, ALS by design:
- требует вычитать весь набор данных в память (для каждого фактора необходимо помнить, у каких объектов он есть и какой имеет вес)
- является последовательным

## Бенчмарки

## Выводы
1) При разработке большое внимание уделялось прозрачности архитектуры, поддерживаемости кода, тестированию. Делать это на Java легко и приятно благодаря богатому набору библиотек, умной среде разработки, ясной объектно-ориентированной модели. Однако при проектировании была допущена ошибка: в изначальной архитектуре проекта для обучения SGD не был предусмотрен паралеллизм, предполагалось добавить его позднее. На практике это привело к необходимости почти полного переписывания процедур для ввода и обучения.

2) SGD just wants to work, этот метод даже с постоянным темпом обучения достаточно легко настраивается и выдаёт адекватные результаты, сопоставимые с известными Open-source инструментам для обучения. Кроме того, он не потребляет много памяти, его можно эффективно распараллелить.

## Инструкция по развёртыванию
### Сборка из исходного кода
