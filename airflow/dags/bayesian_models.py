from sklearn.base import BaseEstimator

# going to do some hand-modelling.

# treat average player score as a fixed hidden variable to be estimated.

# at the start of the season we have no information about how good the player is. Start with a general prior based on the distribution of scores for players in this position.

# each game contributes evidence.

# wait up

# assume no prior

# after one game, player's expected score is their last score. Std dev of this
# is undefined?

# after two games, players expected score is the avg. Std dev is std dev of
# distribution / sqrt(2)

# in general, expected score is normally distributed with mu = mean score,
# sigma = std dev of scores / sqrt(n)

class MeanPointsPredictor(BaseEstimator):
    def fit(X, y):
        pass
    def predict(X):
        return X["points_per_game"]

class BayesianPointsPredictor(BaseEstimator):
    def __init__(self, weight_par=10, prior="global"):
        """Estimator that predicts score as a weighted average of
        some prior expectation and the player's points per game. The weight
        given to points per game is a function of the number of games played
        and the weight_par. Higher weight pars give more weight to the prior.
        
        arguments
        =========
        weight_par: float, must be >= 0.
        prior: str, can be one of "global", "team", "position". The player's
        prior points expectation is set to the global mean score, team mean or
        position mean.
        """
        self.weight_par = weight_par
        self.prior = prior

    def _predict(self, X, weight_par):
        n = X["appearances"]
        weight = n / (n + weight_par)
        return X["points_per_game"] * weight + self.overall_mean * (1 - weight)

    def fit(X, y):
        if self.prior == "global":
            self.overall_mean = y.mean()
        if self.prior == "team":
            # find all teammates
            team_means = y.groupby(X["team_id"]).mean()
            self.overall_mean = team_means.loc[X["team_id"]]
        if self.prior == "position":
            # find all teammates
            team_means = y.groupby(X["position_id"]).mean()
            self.overall_mean = team_means.loc[X["position_id"]]
        # grid search over weight par
        weights = np.logspace(-1, 2, 10)
        scores = []
        for weight in weights:
            preds = _predict(X, weights)
            scores.append(((preds - y)**2).sum())
        scores = pd.Series(scores, index=weights)
        self.weight_par = scores.idxmin()

    def predict(X):
        return self._predict(X, self.weight_par)

