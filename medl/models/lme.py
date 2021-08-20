import numpy as np
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

class MixedGLMClassifier:
    def __init__(self, feature_name, site_name='SITE', label_name='CONVERSION'):
        self.strFeature = feature_name
        self.strSite = site_name

        self.strFormula = f'{label_name} ~ {feature_name}'
        self.dictRandomEffects = {'site_slope': f'0 + C({site_name}):{feature_name}'}
        self.model = None

    def fit(self, dfX):       
        self.model = BinomialBayesMixedGLM.from_formula(self.strFormula, self.dictRandomEffects, dfX)
        self.fit_result = self.model.fit_vb()

    def predict(self, dfX):
        if self.model is None:
            raise UserWarning('Model has not been fit yet.')

        dfRE = self.fit_result.random_effects()

        lsRESlopes = [dfRE['Mean'].filter(like=str(x)).values[0] for x in dfX[self.strSite]]
        arrRESlopes = np.array(lsRESlopes)
        arrRandomEffects = arrRESlopes * dfX[self.strFeature].values

        arrDesignMat = np.ones((dfX.shape[0], 2))
        arrDesignMat[:, 1] = dfX[self.strFeature]

        arrPredFE = self.fit_result.predict(arrDesignMat, linear=True)
        arrPredMixed = arrPredFE + arrRandomEffects

        arrPredMixedLogit = self.model.family.link.inverse(arrPredMixed)

        return arrPredMixedLogit

