# Group-by augmentations model for e-commerce datasets
Following these two articles:
* [Universal Machine Learning approach for targeted marketing campaigns](https://medium.com/tinyclues-vision/universal-machine-learning-approach-for-targeted-marketing-campaigns-b6df27f38532?source=friends_link&sk=fbd15107dce03c11244cb7b48ec7d365)
* [Group-by data augmentation for e-commerce datasets](https://medium.com/tinyclues-vision/group-by-data-augmentation-for-e-commerce-datasets-508d5080c0c6?source=friends_link&sk=59247228eb89dc469320144f503ff337)

We present new model architecture that can be used to make predictions to various targeted campaigns without limitation
on offer selection. All code is reproducible and can be used on other datasets. We use public Movielens+IMDB dataset as 
well as Rees e-commerce dataset to show-case advantages of a new model.

More precisely, to better address cold-start problem and to simplify our workflow we want to have a single model that uses all available features defining an offer. We also performed benchmarks for this single model on different offer definitions.

To follow these benchmarks you need to first download datasets and transform them into needed `tf.Dataset` format:
* [Notebook for Movielens+IMDB dataset](https://github.com/tinyclues/group-by-augmentations-model/blob/master/dataset_preprocessing/movielens%20with%20imdb.ipynb)
* [Notebook for Rees E-commerce dataset](https://github.com/tinyclues/group-by-augmentations-model/blob/master/dataset_preprocessing/rees_ecommerce.ipynb)

Global training structure and performance gap we see for simple model's architecture are presented in
* [Notebook for Movielens+IMDB training of a simple model](https://github.com/tinyclues/group-by-augmentations-model/blob/master/training/movielens%20simple%20model.ipynb)

Proposed new architecture with group-by augmentations is presented in the following notebook:
* [Group-by augmentations model for Movielens-IMDB](https://github.com/tinyclues/group-by-augmentations-model/blob/master/training/movielens%20group%20by%20augmentations.ipynb)

along with benchmarks showing that it fulfills performance gaps seen before.  

We do the same benchmarks on Rees E-commerce dataset following the same templates:
* [Notebook for simple model](https://github.com/tinyclues/group-by-augmentations-model/blob/master/training/rees%20simple%20model.ipynb)
* [Notebook for group-by augmentations model](https://github.com/tinyclues/group-by-augmentations-model/blob/master/training/rees%20group%20by%20augmentations.ipynb)