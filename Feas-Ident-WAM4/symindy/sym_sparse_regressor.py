import logging
import operator
import random
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from deap import base, creator, gp, tools
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

from symindy.library import Library


class SymSparseRegressor:
    def __init__(
        self,
        ngen=5,
        ntrees=5,
        dims=2,
        library_name="generalized",
        sparsity_coef=1.0,
        n_individuals=300,
        max_depth=2,
        mutpb=0.8,
        cxpb=0.7,
        score_metric=r2_score,
        score_metric_kwargs=None,
        regressor=None,
        regressor_kwargs=None,
        nc=0,
        seed=0,
        verbose=False,
        tr_te_ratio=0.8,
        invalid_fitness=-1e12,
        constant_tol=1e-12,
        coef_tol=1e-10,
    ):
        """
        Symbolic sparse regression for general supervised learning:
            y = f(X)

        Parameters
        ----------
        ngen : int
            Number of GP generations.
        ntrees : int
            Number of symbolic trees per individual.
        dims : int
            Number of input dimensions.
        library_name : str
            Primitive library name passed into symindy.library.Library.
        sparsity_coef : float
            Strength of structural sparsity penalty.
        n_individuals : int
            Population size.
        max_depth : int
            Maximum GP tree depth.
        mutpb : float
            Mutation probability.
        cxpb : float
            Crossover probability.
        score_metric : callable
            Metric used on validation predictions. Higher is better.
        score_metric_kwargs : dict or None
            Extra kwargs for score_metric.
        regressor : sklearn-like regressor class or None
            Defaults to sklearn.linear_model.Lasso.
        regressor_kwargs : dict or None
            Kwargs passed to regressor constructor.
        nc : int
            Number of numeric constants in the primitive set.
        seed : int
            Random seed.
        verbose : bool
            Verbose logging.
        tr_te_ratio : float or None
            Train/test split ratio. If None, score on all data.
        invalid_fitness : float
            Fitness assigned to invalid individuals.
        constant_tol : float
            Drop symbolic features with std below this threshold.
        coef_tol : float
            Threshold for determining active regression coefficients.
        """
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format="%(levelname)s %(message)s",
        )

        self.ngen = ngen
        self.ntrees = ntrees
        self.dims = dims
        self.library_name = library_name
        self.sparsity_coef = sparsity_coef
        self.n_individuals = n_individuals
        self.max_depth = max_depth
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.score_metric = score_metric
        self.score_metric_kwargs = {} if score_metric_kwargs is None else score_metric_kwargs
        self.regressor = Lasso if regressor is None else regressor
        self.regressor_kwargs = (
            {"alpha": 1e-3, "fit_intercept": True, "max_iter": 20000}
            if regressor_kwargs is None
            else regressor_kwargs
        )
        self.nc = nc
        self.seed = seed
        self.verbose = verbose
        self.tr_te_ratio = tr_te_ratio
        self.invalid_fitness = invalid_fitness
        self.constant_tol = constant_tol
        self.coef_tol = coef_tol

        if self.verbose:
            logging.info(self.__class__.__name__ + "__init__")

    def configure_DEAP(self, ntrees=5, nc=0, dimensions=2, max_depth=2):
        if self.verbose:
            logging.info(self.__class__.__name__ + " configure_DEAP")

        def _random_mating_operator(ind1, ind2):
            roll = random.random()
            if roll < 0.5:
                return gp.cxOnePoint(ind1, ind2)
            else:
                return gp.cxOnePointLeafBiased(ind1, ind2, termpb=0.5)

        def _random_mutation_operator(individual):
            roll = random.random()
            if roll < 0.5:
                return gp.mutInsert(individual, pset=pset)
            elif roll < 0.66:
                return gp.mutShrink(individual)
            else:
                return gp.mutNodeReplacement(individual, pset=pset)

        def _rename_args(pset, nc, dimensions):
            argnames = {}
            for dim in range(dimensions):
                argnames[f"ARG{dim}"] = f"x{dim}"
            for i in range(nc):
                argnames[f"ARG{i + dimensions}"] = f"c{i}"
            pset.renameArguments(**argnames)
            return pset

        def _create_toolbox(pset, ntrees, max_depth=2):
            from deap import creator

            # avoid creator redefinition crashes in notebooks
            if not hasattr(creator, "FitnessMax"):
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            if not hasattr(creator, "Subindividual"):
                creator.create("Subindividual", gp.PrimitiveTree)
            if not hasattr(creator, "Individual"):
                creator.create("Individual", list, fitness=creator.FitnessMax)

            toolbox = base.Toolbox()
            toolbox.register(
                "expr",
                gp.genHalfAndHalf,
                pset=pset,
                type_=pset.ret,
                min_=0,
                max_=max_depth,
            )
            toolbox.register(
                "subindividual",
                tools.initIterate,
                creator.Subindividual,
                toolbox.expr,
            )
            toolbox.register(
                "individual",
                tools.initRepeat,
                creator.Individual,
                toolbox.subindividual,
                n=ntrees,
            )
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("compile", gp.compile, pset=pset)
            toolbox.register("select", tools.selTournament, tournsize=2)
            toolbox.register("mate", _random_mating_operator)
            toolbox.register("mutate", _random_mutation_operator)

            history = tools.History()
            toolbox.decorate("mate", history.decorator)
            toolbox.decorate("mutate", history.decorator)
            toolbox.decorate(
                "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth)
            )
            toolbox.decorate(
                "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth)
            )
            return toolbox, creator, history

        self.max_depth = max_depth
        lib = Library(nc, dimensions, library_name=self.library_name)
        pset = lib()
        pset = _rename_args(pset, nc, dimensions)
        toolbox, creator, history = _create_toolbox(pset, ntrees, max_depth=self.max_depth)
        return toolbox, creator, pset, history

    @staticmethod
    def _split_array(arr, ratio):
        if arr is None:
            return None, None
        idx = int(ratio * len(arr))
        return arr[:idx], arr[idx:]

    @staticmethod
    def _ensure_2d_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    @staticmethod
    def _ensure_1d_y(y):
        y = np.asarray(y, dtype=float).ravel()
        return y

    @staticmethod
    def evalSymbolicReg(
        individual,
        ntrees,
        max_depth,
        toolbox,
        x_train,
        y_train,
        score_metric=None,
        score_metric_kwargs=None,
        regressor=None,
        regressor_kwargs=None,
        flag_solution=False,
        tr_te_ratio=0.8,
        sparsity_coef=1.0,
        invalid_fitness=-1e12,
        constant_tol=1e-12,
        coef_tol=1e-10,
    ):
        """
        Evaluate one GP individual for supervised symbolic sparse regression.

        Returns
        -------
        [fitness] if flag_solution is False
        result dict otherwise
        """
        if score_metric is None:
            score_metric = r2_score
        if score_metric_kwargs is None:
            score_metric_kwargs = {}
        if regressor is None:
            regressor = Lasso
        if regressor_kwargs is None:
            regressor_kwargs = {"alpha": 1e-3, "fit_intercept": True, "max_iter": 20000}

        def validate_input(X, y):
            if X is None or y is None:
                raise ValueError("x_train and y_train must both be provided.")
            if len(X) < 3:
                raise ValueError("Need at least 3 samples.")
            if len(X) != len(y):
                raise ValueError("x_train and y_train must have same number of rows.")
            if X.ndim != 2:
                raise ValueError("x_train must be 2D with shape (N, d).")

        def build_feature_matrix(individual, toolbox, X):
            funcs = [toolbox.compile(expr=individual[i]) for i in range(ntrees)]

            cols = []
            valid_tree_indices = []

            for i, fn in enumerate(funcs):
                try:
                    col = np.array([fn(*row) for row in X], dtype=float).ravel()
                except Exception:
                    continue

                if col.shape[0] != X.shape[0]:
                    continue
                if not np.all(np.isfinite(col)):
                    continue
                if np.std(col) < constant_tol:
                    continue

                cols.append(col)
                valid_tree_indices.append(i)

            if len(cols) == 0:
                return None, []

            Theta = np.column_stack(cols)
            return Theta, valid_tree_indices

        X = SymSparseRegressor._ensure_2d_X(x_train)
        y = SymSparseRegressor._ensure_1d_y(y_train)
        validate_input(X, y)

        if tr_te_ratio is not None:
            X_tr, X_te = SymSparseRegressor._split_array(X, tr_te_ratio)
            y_tr, y_te = SymSparseRegressor._split_array(y, tr_te_ratio)
        else:
            X_tr, X_te = X, X
            y_tr, y_te = y, y

        Theta_tr, valid_idx = build_feature_matrix(individual, toolbox, X_tr)
        if Theta_tr is None or Theta_tr.shape[1] == 0:
            return {"fitness": invalid_fitness} if flag_solution else [invalid_fitness]

        Theta_te, _ = build_feature_matrix(individual, toolbox, X_te)
        if Theta_te is None:
            return {"fitness": invalid_fitness} if flag_solution else [invalid_fitness]
        if Theta_te.shape[1] != Theta_tr.shape[1]:
            return {"fitness": invalid_fitness} if flag_solution else [invalid_fitness]

        try:
            reg_model = regressor(**regressor_kwargs)
            reg_model.fit(Theta_tr, y_tr)
            y_pred = reg_model.predict(Theta_te)
            fitness = score_metric(y_te, y_pred, **score_metric_kwargs)
        except Exception:
            return {"fitness": invalid_fitness} if flag_solution else [invalid_fitness]

        coef = np.asarray(reg_model.coef_).ravel()
        active_mask = np.abs(coef) > coef_tol

        n_nodes = 0
        for k, tree_idx in enumerate(valid_idx):
            if k < len(active_mask) and active_mask[k]:
                n_nodes += len(individual[tree_idx])

        max_nnodes = (2 ** (1 + max_depth)) * ntrees
        fitness -= sparsity_coef * (n_nodes / max_nnodes)

        result = {
            "fitness": fitness,
            "reg_model": reg_model,
            "valid_idx": valid_idx,
            "coef": coef,
            "intercept": getattr(reg_model, "intercept_", 0.0),
            "individual": individual,
            "Theta_tr_shape": Theta_tr.shape,
            "Theta_te_shape": Theta_te.shape,
        }

        return result if flag_solution else [fitness]

    @staticmethod
    def my_eaSimple(
        population,
        toolbox_local,
        cxpb,
        mutpb,
        ngen,
        ntrees,
        stats=None,
        halloffame=None,
        verbose=False,
    ):
        def _my_varAnd(population, toolbox_local, cxpb, mutpb):
            offspring = [toolbox_local.clone(ind) for ind in population]

            for i in range(1, len(offspring), 2):
                if random.random() < cxpb:
                    h_component = random.randint(0, ntrees - 1)
                    (
                        offspring[i - 1][h_component],
                        offspring[i][h_component],
                    ) = toolbox_local.mate(
                        offspring[i - 1][h_component],
                        offspring[i][h_component],
                    )
                    del offspring[i - 1].fitness.values, offspring[i].fitness.values

            for i in range(len(offspring)):
                for h_component in range(ntrees):
                    if random.random() < mutpb:
                        (offspring[i][h_component],) = toolbox_local.mutate(
                            offspring[i][h_component]
                        )
                        del offspring[i].fitness.values

            return offspring

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox_local.map(toolbox_local.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            if verbose:
                print("Fitness:", fit)

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)

        for gen in range(1, ngen + 1):
            offspring = toolbox_local.select(population, len(population))
            offspring = _my_varAnd(offspring, toolbox_local, cxpb, mutpb)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox_local.map(toolbox_local.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            if halloffame is not None:
                halloffame.update(offspring)

            population[:] = offspring

            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            if verbose:
                print(logbook.stream)
                if halloffame is not None and len(halloffame) > 0:
                    for i in range(ntrees):
                        print(halloffame[0][i])

        return population, logbook, halloffame

    @staticmethod
    def init_stats():
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        return mstats

    def _build_theta_from_best(self, X):
        X = self._ensure_2d_X(X)
        individual = self.hof[0]
        funcs = [self.toolbox.compile(expr=individual[i]) for i in range(self.ntrees)]

        cols = []
        for i in self.final_model["valid_idx"]:
            fn = funcs[i]
            col = np.array([fn(*row) for row in X], dtype=float).ravel()
            cols.append(col)

        if len(cols) == 0:
            raise ValueError("No valid features exist in the fitted model.")

        return np.column_stack(cols)

    def fit(self, x_train, y_train):
        if self.verbose:
            logging.info(self.__class__.__name__ + " fit")

        random.seed(self.seed)
        np.random.seed(self.seed)

        x_train = self._ensure_2d_X(x_train)
        y_train = self._ensure_1d_y(y_train)

        if x_train.shape[1] != self.dims:
            raise ValueError(
                f"x_train has {x_train.shape[1]} columns but dims={self.dims}."
            )
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("x_train and y_train must have the same number of rows.")

        toolbox, creator, pset, history = self.configure_DEAP(
            ntrees=self.ntrees,
            nc=self.nc,
            dimensions=self.dims,
            max_depth=self.max_depth,
        )

        toolbox.register(
            "evaluate",
            self.evalSymbolicReg,
            ntrees=self.ntrees,
            max_depth=self.max_depth,
            toolbox=toolbox,
            x_train=x_train,
            y_train=y_train,
            score_metric=self.score_metric,
            score_metric_kwargs=self.score_metric_kwargs,
            regressor=self.regressor,
            regressor_kwargs=self.regressor_kwargs,
            flag_solution=False,
            tr_te_ratio=self.tr_te_ratio,
            sparsity_coef=self.sparsity_coef,
            invalid_fitness=self.invalid_fitness,
            constant_tol=self.constant_tol,
            coef_tol=self.coef_tol,
        )

        toolbox.register(
            "retrieve_model",
            self.evalSymbolicReg,
            ntrees=self.ntrees,
            max_depth=self.max_depth,
            toolbox=toolbox,
            x_train=x_train,
            y_train=y_train,
            score_metric=self.score_metric,
            score_metric_kwargs=self.score_metric_kwargs,
            regressor=self.regressor,
            regressor_kwargs=self.regressor_kwargs,
            flag_solution=True,
            tr_te_ratio=self.tr_te_ratio,
            sparsity_coef=self.sparsity_coef,
            invalid_fitness=self.invalid_fitness,
            constant_tol=self.constant_tol,
            coef_tol=self.coef_tol,
        )

        mstats = self.init_stats()
        pop = toolbox.population(n=self.n_individuals)
        hof_ = tools.HallOfFame(1)

        pop, log, hof = self.my_eaSimple(
            pop,
            toolbox,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            ngen=self.ngen,
            ntrees=self.ntrees,
            stats=mstats,
            halloffame=hof_,
            verbose=self.verbose,
        )

        final_model = toolbox.retrieve_model(hof[0])
        if final_model is None or final_model.get("fitness", self.invalid_fitness) <= self.invalid_fitness:
            raise RuntimeError("Best individual did not produce a valid regression model.")

        self.x_train = x_train
        self.y_train = y_train

        self.toolbox = toolbox
        self.creator = creator
        self.pset = pset
        self.history = history
        self.population = pop
        self.log = log
        self.hof = hof
        self.final_model = final_model

        print("\nEstimated symbolic features:")
        for i in range(self.ntrees):
            print(f"f{i}: {hof[0][i]}")

        print("\nSelected sparse regression terms:")
        coef = self.final_model["coef"]
        valid_idx = self.final_model["valid_idx"]
        intercept = self.final_model["intercept"]

        for k, tree_idx in enumerate(valid_idx):
            if k < len(coef) and abs(coef[k]) > self.coef_tol:
                print(f"{coef[k]: .6f} * f{tree_idx}")

        print(f"Intercept: {float(np.asarray(intercept)):.6f}")
        print(f"Fitness: {self.final_model['fitness']:.6f}\n")

        return self

    def predict(self, X):
        if self.verbose:
            logging.info(self.__class__.__name__ + " predict")

        Theta = self._build_theta_from_best(X)
        return self.final_model["reg_model"].predict(Theta)

    def score(self, X, y, metric=None, metric_kwargs=None):
        if self.verbose:
            logging.info(self.__class__.__name__ + " score")

        if metric is None:
            metric = self.score_metric
        if metric_kwargs is None:
            metric_kwargs = {}

        y = self._ensure_1d_y(y)
        y_pred = self.predict(X)
        return metric(y, y_pred, **metric_kwargs)

    def plot_trees(self, show=False):
        if self.verbose:
            logging.info(self.__class__.__name__ + " plot_trees")

        try:
            expr = self.hof[0]
            nrows = max(1, int(np.floor(self.ntrees / 2)))
            ncols = int(np.ceil(self.ntrees / max(nrows, 1)))

            fig, axs = plt.subplots(nrows, ncols, figsize=(16, 9))
            axs = np.atleast_1d(axs).ravel()

            for i, ax in zip(range(self.ntrees), axs):
                nodes, edges, labels = gp.graph(expr[i])
                g = nx.Graph()
                g.add_nodes_from(nodes)
                g.add_edges_from(edges)
                pos = nx.nx_agraph.pygraphviz_layout(g, prog="dot")
                nx.draw(
                    g,
                    pos,
                    with_labels=True,
                    ax=ax,
                    labels=labels,
                    node_color="#99CCFF",
                    edge_color="k",
                    font_size=20,
                    font_color="k",
                )
                ax.set_axis_off()

            plt.margins(0.2)
            plt.axis("off")
            plt.tight_layout()

            if show:
                plt.show()

            return fig, ax
        except Exception as exe:
            logging.error(
                "Failed plotting the trees. Most likely pygraphviz is missing. "
                f"Original Error: {exe}"
            )
            return None, None


if __name__ == "__main__":
    pass