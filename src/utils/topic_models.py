import os
from typing import Tuple, Union

import anndata  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import mira  # type: ignore[import-untyped]
import torch
from mira.topic_model.modality_mixins.accessibility_model import AccessibilityModel  # type: ignore[import-untyped]
from mira.topic_model.modality_mixins.expression_model import ExpressionModel  # type: ignore[import-untyped]
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_model_modality(model) -> str:
    """Return 'expression' or 'accessibility' based on model type."""
    if isinstance(model, ExpressionModel):
        return "expression"
    elif isinstance(model, AccessibilityModel):
        return "accessibility"
    else:
        raise TypeError(f"Model of type '{type(model)}' is not a dirichlet expression or accessibility object")


def load_or_create_mira_expression_topic_model(
    rna_adata: anndata.AnnData, 
    model_path: Union[None, str] = None
    ) -> ExpressionModel:
    
    try:
        return mira.topic_model.load_model(model_path)
    except ValueError:
        logging.info(f"No MIRA model.pth file found at path {model_path}, creating...")

    logging.info("Creating MIRA expression model from RNAseq AnnData")
    model: ExpressionModel = mira.topics.make_model(
        rna_adata.n_obs, rna_adata.n_vars,
        feature_type = 'expression',
        highly_variable_key='highly_variable',
        counts_layer='counts',
        )
    
    if model_path:
        try:
            model.save(model_path)
        except OSError:
            logging.error(f"Model save path {model_path} is not valid, model ill not be saved!")
    else:
        logging.warning(f"Warning: model_path variable is {model_path}, the model wont be saved at this point. \
            The model will be saved after topic training")
    
    return model

def load_or_create_mira_accessibility_topic_model(
    atac_adata: anndata.AnnData,
    model_path: Union[None,str] = None
    ) -> AccessibilityModel:
    
    try:
        return mira.topic_model.load_model(model_path)
    except ValueError:
        logging.info(f"No MIRA model.pth file found at path {model_path}, creating...")
    
    if torch.cuda.is_available():
        atac_encoder = "DAN"
    else:
        atac_encoder = "light"

    logging.info("Creating MIRA accessibility model from ATACseq AnnData")
    model: AccessibilityModel = mira.topics.make_model(
        *atac_adata.shape,
        feature_type = 'accessibility',
        endogenous_key='endogenous_peaks', # which peaks are used by the encoder network
        atac_encoder=atac_encoder
    )
    
    if model_path:
        try:
            model.save(model_path)
        except OSError:
            logging.error(f"Model save path {model_path} is not valid, model ill not be saved!")
    else:
        logging.warning(f"Warning: model_path variable is {model_path}, the model will not be saved at this point. \
            Next save point is after topic training")
    
    return model

def set_model_learning_parameters(
    model: Union[ExpressionModel, AccessibilityModel], 
    adata: Union[anndata.AnnData, str],
    fig_dir: str = 'figures'
    ) -> Tuple[Union[ExpressionModel, AccessibilityModel], int]:
    
    model_type = get_model_modality(model)
    assert model_type == "accessibility" or "expression"

    logging.info("Running the learning rate test:")
    min_lr, max_lr = model.get_learning_rate_bounds(adata)
    
    learn_rate_ax = model.plot_learning_rate_bounds()

    learn_rate_fig = learn_rate_ax.get_figure()

    if isinstance(learn_rate_fig, plt.Figure):
        learn_rate_fig.savefig(
        os.path.join(fig_dir, f"{model_type}_learning_rate_bounds.png"),
        dpi=200,
        bbox_inches='tight'
    )

    logging.info(f"Setting min learning rate to {min_lr} and max learning rate to {max_lr}")
    model.set_learning_rates(min_lr, max_lr) # for larger datasets, the default of 1e-3, 0.1 usually works well.

    topic_contributions = mira.topics.gradient_tune(model, adata)

    sig_topic_contributions = [x for x in topic_contributions if x > 0.05]

    num_sig_topics: int = len(sig_topic_contributions)
    
    topic_contrib_ax: plt.Axes = mira.pl.plot_topic_contributions(topic_contributions, num_sig_topics)
    
    topic_contrib_fig = topic_contrib_ax.get_figure()
    
    if isinstance(topic_contrib_fig, plt.Figure):
        topic_contrib_fig.savefig(
        os.path.join(fig_dir, f"{model_type}_topic_contributions.png"),
        dpi=200,
        bbox_inches='tight'
    )

    return model, num_sig_topics

def create_and_fit_bayesian_tuner_to_data(
    model: Union[ExpressionModel, AccessibilityModel], 
    adata: anndata.AnnData, 
    num_sig_topics: int, 
    tuner_save_name: str,
    model_save_path: str,
    n_jobs: int = 5,
    fig_dir: str = 'figures',
    plot_loss: bool = True,
    plot_pareto: bool = True
    ) -> Union[ExpressionModel, AccessibilityModel]:
    """
    Create a mira.topics.BayesianTuner model and fit it to the data. Saves 
    the tuner model to 'save_name'. Plots the training loss values over time
    and the 

    Args:
        model (ExpressionModel | AccessibilityModel): 
            MIRA model built from the AnnData object.
        adata (anndata.AnnData): 
            AnnData expression of accessility object used to build the model.
        num_sig_topics (int): 
            Number of topics used to represent the data. Build using the `set_model_learning_parameters()` 
            function.
        tuner_save_name (str): 
            Directory path to save the BayesianTuner model to. Saves the model to the `models/<save_name>`
            directory.
        model_save_name (str):
            Full path for saving the trained model file.
        n_jobs (int):
            Number of parallelization jobs. Max is 5 unless using the REDIS backend
        fig_dir (str):
            Directory to save figures to. (Default = `figures`)
        plot_loss (bool): 
            Specifies whether to plot the loss over time. saves to the `figures` directory (Default = True)
        plot_pareto (bool): 
            Set as `True` to plot the pareto front. Losses should be convex with respect to the number 
            of topics. Ensures that a reasonable number of topics was chosen for the model and that
            the tuner converged on that estimate.


    Returns:
        model (ExpressionModel | AccessibilityModel): MIRA model containing the best model weights
    """
    model_type = get_model_modality(model)
    
    if n_jobs > 5:
        raise ValueError(
            f"REDIS backend not supported in this function, n_jobs must be less than 5. \
            Currently set to {n_jobs}"
            )
        
    logging.info("Creating Bayesian tuner object")
    tuner = mira.topics.BayesianTuner(
            model = model,
            n_jobs=n_jobs,
            save_name = tuner_save_name,
            log_steps=False
            #### IMPORTANT
            min_topics = max(3, num_sig_topics - 5), max_topics = min(50, num_sig_topics + 20), # tailor for your dataset!!!!
            #### See "Notes on min_topics, max_topics" above
            #storage = mira.topics.Redis() # if using REDIS backend for more (>5) processes
    )

    logging.info("Fitting the data to the tuner")
    tuner.fit(adata)

    if plot_loss:
        loss_ax = tuner.plot_intermediate_values(palette='Spectral_r',
                                        log_hue=True, figsize=(7,3))
        # intermed_val_plt_ax.set(ylim = (7e2, 7.7e2))

        loss_fig = loss_ax.get_figure()

        if isinstance(loss_fig, plt.Figure):
            loss_fig.savefig(
                os.path.join(fig_dir, f"{model_type}_tuner_intermediate_loss_values.png"), 
                dpi=200,
                bbox_inches='tight'
                )

    if plot_pareto:
        pareto_plt_ax = tuner.plot_pareto_front(include_pruned_trials=False, label_pareto_front=True,
                            figsize = (8,8))

        pareto_fig = pareto_plt_ax.get_figure()

        if isinstance(pareto_fig, plt.Figure):
            pareto_fig.savefig(
                os.path.join(fig_dir, f"{model_type}_check_pareto_front_convex_losses.png"), 
                dpi=200,
                bbox_inches='tight'
                )

    logging.info("Finding best model")
    model = tuner.fetch_best_weights()

    logging.info(f"Saving best model to '{model_save_path}'")
    try:
        model.save(model_save_path)
    except OSError:
        logging.error(f"Model save path {model_save_path} is not valid, model will not be saved!")
    
    return model