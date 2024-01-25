"""
Utilities for various UCI repository datasets.

This is a public data source and no special action is required
to access it.

For more information on datasets and access in TableShift, see:
* https://tableshift.org/datasets.html
* https://github.com/mlfoundations/tableshift

"""
import pandas as pd

from tableshift.core.features import FeatureList, Feature, cat_dtype

IRIS_FEATURES = FeatureList(features=[
    Feature('Sepal_Length', float, name_extended='sepal length'),
    Feature('Sepal_Width', float, name_extended='sepal width'),
    Feature('Petal_Length', float, name_extended='petal length'),
    Feature('Petal_Width', float, name_extended='petal width'),
    Feature('Class', cat_dtype, is_target=True),
], documentation="https://archive.ics.uci.edu/ml/datasets/Iris")

DRY_BEAN_FEATURES = FeatureList(features=[
    Feature('Area', int, name_extended='area of bean zone in pixels'),
    Feature('Perimeter', float, name_extended='circumference of bean'),
    Feature('MajorAxisLength', float,
            name_extended='distance between the ends of the longest line that can be drawn from the bean'),
    Feature('MinorAxisLength', float,
            name_extended='length of longest line that can be drawn from the bean while standing perpendicular to the main axis'),
    Feature('AspectRation', float,
            name_extended='ratio between major and minor axis'),
    Feature('Eccentricity', float,
            name_extended='eccentricity of the ellipse having the same moments as the region'),
    Feature('ConvexArea', int,
            name_extended='area of the smallest convex polygon that can contain the area of a bean seed in pixels'),
    Feature('EquivDiameter', float,
            name_extended='diameter of a circle having the same area as a bean seed area'),
    Feature('Extent', float,
            name_extended='ratio of the pixels in the bounding box to the bean area'),
    Feature('Solidity', float,
            name_extended='ratio of the pixels in the convex shell to those found in beans'),
    Feature('roundness', float),
    Feature('Compactness', float),
    Feature('ShapeFactor1', float, name_extended='shape factor 1'),
    Feature('ShapeFactor2', float, name_extended='shape factor 2'),
    Feature('ShapeFactor3', float, name_extended='shape factor 3'),
    Feature('ShapeFactor4', float, name_extended='shape factor 4'),
    Feature('Class', cat_dtype, name_extended='class of bean', is_target=True),
], documentation="https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset")

HEART_DISEASE_FEATURES = FeatureList(features=[
    Feature('age', float, name_extended='age in years'),
    Feature('sex', float, value_mapping={1.: 'male', 0.: 'female'}),
    Feature('cp', float, name_extended='chest pain type',
            value_mapping={1.: 'typical angina',
                           2.: 'atypical angina',
                           3.: 'non-anginal pain',
                           4.: 'asymptomatic'}),
    Feature('trestbps', float,
            name_extended='resting blood pressure (in mm Hg on admission to the hospital)'),
    Feature('chol', float, name_extended='serum cholesterol in mg/dl'),
    Feature('fbs', float,
            name_extended='fasting blood sugar > 120 mg/dl)',
            value_mapping={1.: 'true', 0.: 'false'}),
    Feature('restecg', float,
            name_extended='resting electrocardiographic results',
            value_mapping={
                0.: 'normal',
                1.: 'having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)',
                2.: "showing probable or definite left ventricular hypertrophy by Estes' criteria"
            }),
    Feature('thalach', float, name_extended='maximum heart rate achieved'),
    Feature('exang', float, name_extended='exercise induced angina',
            value_mapping={1.: 'yes', 0.: 'no'}),
    Feature('oldpeak', float,
            name_extended='ST depression induced by exercise relative to rest'),
    Feature('slope', float,
            name_extended='the slope of the peak exercise ST segment',
            value_mapping={1.: 'upsloping', 2.: 'flat', 3.: 'downsloping'}),
    Feature('ca', float,
            name_extended='number of major vessels colored by flourosopy'),
    Feature('thal', float,
            value_mapping={
                3.: 'normal', 6.: 'fixed defect', 7.: 'reversible defect'}),
    Feature('num', float, is_target=True,
            name_extended='diagnosis of heart disease (angiographic disease status)',
            value_mapping={
                0.: 'none',
                1.: 'minimal',
                2.: 'mild',
                3.: 'moderate',
                4.: 'severe'
            }),
], documentation="https://archive.ics.uci.edu/ml/datasets/Heart+Disease")

WINE_CULTIVARS_FEATURES = FeatureList(features=[
    Feature('Cultivars', int, is_target=True, name_extended='cultivar'),
    Feature('Alcohol', float),
    Feature('Malic acid', float),
    Feature('Ash', float),
    Feature('Alcalinity of ash', float, name_extended='alkalinity of ash'),
    Feature('Magnesium', float),
    Feature('Total Phenols', float),
    Feature('Flavanoids', float),
    Feature('Nonflavanoid phenols', float),
    Feature('Proanthocyanins', float),
    Feature('Color intensity', float),
    Feature('Hue', float),
    Feature('OD280/OD315 of diluted wines', float),
    Feature('Proline', float),
], documentation="https://archive.ics.uci.edu/ml/datasets/Wine")

WINE_QUALITY_FEATURES = FeatureList(features=[
    Feature('fixed acidity', float),
    Feature('volatile acidity', float),
    Feature('citric acid', float),
    Feature('residual sugar', float),
    Feature('chlorides', float),
    Feature('free sulfur dioxide', float),
    Feature('total sulfur dioxide', float),
    Feature('density', float),
    Feature('pH', float),
    Feature('sulphates', float),
    Feature('alcohol', float),
    Feature('quality', int, is_target=True),
    Feature('red_or_white', cat_dtype, name_extended="red or white wine"),
], documentation="https://archive.ics.uci.edu/ml/datasets/Wine+Quality")

RICE_FEATURES = FeatureList(features=[
    Feature('Area', int, name_extended='area of rice grain in pixels'),
    Feature('Perimeter', float, name_extended='circumference of rice grain'),
    Feature('Major_Axis_Length', float,
            name_extended='distance between the ends of the longest line that can be drawn from the rice grain'),
    Feature('Minor_Axis_Length', float,
            name_extended='length of longest line that can be drawn from the rice grain while standing perpendicular to the main axis'),
    Feature('Eccentricity', float,
            name_extended='eccentricity of the ellipse having the same moments as the region'),
    Feature('Convex_Area', int,
            name_extended='area of the smallest convex polygon that can contain the area of a rice grain in pixels'),
    Feature('Extent', float,
            name_extended='ratio of the pixels in the bounding box to the rice grain area'),
    Feature('Class', cat_dtype, is_target=True, name_extended='class of rice'),
],
    documentation="https://www.kaggle.com/datasets/muratkokludataset/rice-dataset-commeo-and-osmancik")

BREAST_CANCER_FEATURES = FeatureList(features=[
    Feature("id", int, name_extended="id number"),
    Feature("diagnosis", cat_dtype, is_target=True,
            value_mapping={'M': 'malignant', 'B': 'benign'}),
    Feature("radius_mean", float,
            name_extended="mean of radius (distances from center to points on the perimeter)"),
    Feature("texture_mean", float,
            name_extended='mean of texture (standard deviation of gray-scale values)'),
    Feature("perimeter_mean", float, name_extended='mean of perimeter'),
    Feature("area_mean", float, name_extended='mean of area'),
    Feature("smoothness_mean", float,
            name_extended='mean of smoothness (local variation in radius lengths)'),
    Feature("compactness_mean", float,
            name_extended='mean of compactness (perimeter^2 / area - 1.0)'),
    Feature("concavity_mean", float,
            name_extended='mean of concavity (severity of concave portions of the contour)'),
    Feature("concave_points_mean", float,
            name_extended='mean of concave points (number of concave portions of the contour)'),
    Feature("symmetry_mean", float, name_extended='mean of symmetry'),
    Feature("fractal_dimension_mean", float,
            name_extended='mean of fractal dimension ("coastline approximation"-1)'),
    Feature("radius_std", float, name_extended='standard deviation of radius'),
    Feature("texture_std", float,
            name_extended='standard deviation of texture'),
    Feature("perimeter_std", float,
            name_extended='standard deviation of perimeter'),
    Feature("area_std", float, name_extended='standard deviation of area'),
    Feature("smoothness_std", float,
            name_extended='standard deviation of smoothness'),
    Feature("compactness_std", float,
            name_extended='standard deviation of compactness'),
    Feature("concavity_std", float,
            name_extended='standard deviation of concavity'),
    Feature("concave_points_std", float,
            name_extended='standard deviation of concave points'),
    Feature("symmetry_std", float,
            name_extended='standard deviation of symmetry'),
    Feature("fractal_dimension_std", float,
            name_extended='standard deviation of fractal dimension'),
    Feature("radius_worst", float, name_extended='worst value of radius'),
    Feature("texture_worst", float,
            name_extended='worst value of texture'),
    Feature("perimeter_worst", float, name_extended='worst value of perimeter'),
    Feature("area_worst", float, name_extended='worst value of area'),
    Feature("smoothness_worst", float,
            name_extended='worst value of smoothness'),
    Feature("compactness_worst", float,
            name_extended='worst value of compactness'),
    Feature("concavity_worst", float, name_extended='worst value of concavity'),
    Feature("concave_points_worst", float,
            name_extended='worst value of concave points'),
    Feature("symmetry_worst", float, name_extended='worst value of symmetry'),
    Feature("fractal_dimension_worst", float,
            name_extended='worst value of fractal dimension'),
],
    documentation='https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29')

CAR_FEATURES = FeatureList(features=[
    Feature('buying', cat_dtype, name_extended='buying price'),
    Feature('maint', cat_dtype, name_extended='maintenance price'),
    Feature('doors', cat_dtype, name_extended='number of doors'),
    Feature('persons', cat_dtype, name_extended='capacity in terms of persons to carry'),
    Feature('lug_boot', cat_dtype,
            name_extended='the size of luggage boot'),
    Feature('safety', cat_dtype, name_extended='estimated safety level of the car'),
    Feature('class', cat_dtype, is_target=True,
            name_extended='car acceptability',
            value_mapping={
                'unacc': 'unacceptable',
                'acc': 'acceptable',
                'good': 'good',
                'vgood': 'very good',
            }),
], documentation='https://archive.ics.uci.edu/ml/datasets/Car+Evaluation')

RAISIN_FEATURES = FeatureList(features=[
    Feature('Area', int, name_extended='area of raisin in pixels'),
    Feature('Perimeter', float, name_extended='circumference of raisin'),
    Feature('MajorAxisLength', float,
            name_extended='distance between the ends of the longest line that can be drawn from the raisin'),
    Feature('MinorAxisLength', float,
            name_extended='length of longest line that can be drawn from the raisin while standing perpendicular to the main axis'),
    Feature('Eccentricity', float,
            name_extended='eccentricity of the ellipse having the same moments as the region'),
    Feature('ConvexArea', int,
            name_extended='area of the smallest convex polygon that can contain the area of a raisin in pixels'),
    Feature('Extent', float,
            name_extended='ratio of the pixels in the bounding box to the raisin area'),
    Feature('Class', cat_dtype, is_target=True,
            name_extended='class of raisin'),
],
    documentation="https://archive.ics.uci.edu/ml/datasets/Raisin+Dataset")

ABALONE_FEATURES = FeatureList(features=[
    Feature('Sex', cat_dtype,
            value_mapping={'M': 'male', 'F': 'female', 'I': 'infant'}),
    Feature('Length', float, name_extended='Longest shell measurement in mm'),
    Feature('Diameter', float,
            name_extended='diameter perpendicular to length in mm'),
    Feature('Height', float, name_extended='with meat in shell in mm'),
    Feature('Whole weight', float, name_extended='whole abalone in grams'),
    Feature('Shucked weight', float, name_extended='weight of meat in grams'),
    Feature('Viscera weight', float,
            name_extended='gut weight in grams after bleeding)'),
    Feature('Shell weight', float,
            name_extended='shell weight in grams after being dried'),
    Feature('Rings', int, 'number of rings', is_target=True),
], documentation='https://archive.ics.uci.edu/ml/datasets/Abalone')


def preprocess_abalone(df: pd.DataFrame) -> pd.DataFrame:
    df['Rings'] = df['Rings'].astype(int)
    return df
