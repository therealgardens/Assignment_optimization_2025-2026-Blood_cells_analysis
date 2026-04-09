# Files Distribution
The **methods** folder will have all the required code for the implementation of the algorithms. The **utils** folder inside can be used to make code that is related to the methods but that is not mainly a part of the algorithm.

The **data_processing** folder is for putting all the code related to handling the data.

The  **data** folder is for putting all the data (both raw and processed versions of it).

**NOTE:** naming convention for everything: *snake_case*

## Handling inner data
Every interchange of the data (dataset or randomly generated) between different parts of the code should be done only through *Numpy* objects, not Pandas or native objects like lists or dictionaries.

This means that for example developed methods should receive numpy objects as input, and their output as well should be numpy objects.

# Branching
Each part of the project will have its own branch:
- feat/gradient_descent
- feat/bcgd_randomized
- feat/bcgd_gauss_southwell
- feat/dataset

**NOTE:** Do not modify files that are not related to other parts of the project in non-related branches