import numpy as np
from sklearn.preprocessing import OneHotEncoder

dictVisit2Months = {'bl': 0,
                    'sc': 0,
                    'm0': 0,
                    'm03': 3,
                    'm06': 6,
                    'm12': 12,
                    'y1': 12,
                    'm18': 18,
                    'm24': 24,
                    'm30': 30,
                    'm36': 36,
                    'm42': 42,
                    'm48': 48,
                    'm54': 54,
                    'm60': 60,
                    'm66': 66,
                    'm72': 72,
                    'm78': 78,
                    'm84': 84,
                    'm90': 90,
                    'm96': 96,
                    'm102': 102,
                    'm108': 108,
                    'm114': 114,
                    'm120': 120,
                    'm126': 126,
                    'm132': 132,
                    'm144': 144,
                    'm156': 156,
                    'm168': 168,
                    'm180': 180
                    }

def convert_adnimerge_categorical(dfDataIn):
    '''
    Convert categorical features from the ADNIMERGE table into one-hot features
    '''
    dfData = dfDataIn.copy()
    dfGender = dfData.pop('PTGENDER')
    dfData['MALE'] = dfGender == 'Male'
    dfEthnicity = dfData.pop('PTETHCAT')
    dfData['HISP/LATINO'] = dfEthnicity == 'Hisp/Latino'
    onehot = OneHotEncoder(sparse=False, drop=['White'])
    dfRace = dfData.pop('PTRACCAT')
    arrRace = onehot.fit_transform(dfRace.values.reshape(-1, 1))
    arrRaceCats = onehot.categories_[0]
    arrRaceCats = np.delete(arrRaceCats, onehot.drop_idx_[0])
    for i, strRace in enumerate(arrRaceCats):
        dfData['RACE-' + strRace.upper()] = arrRace[:, i]
    onehot = OneHotEncoder(sparse=False, drop=['Married'])
    dfMarriage = dfData.pop('PTMARRY')
    arrMarriage = onehot.fit_transform(dfMarriage.values.reshape(-1, 1))
    arrMarriageCats = onehot.categories_[0]
    arrMarriageCats = np.delete(arrMarriageCats, onehot.drop_idx_[0])
    for i, strMarriage in enumerate(arrMarriageCats):
        dfData['MARRIAGE-' + strMarriage.upper()] = arrMarriage[:, i]                    
    return dfData