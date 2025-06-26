## Tutorials
Step 1: [MIRA Tutorial - RNA Topic Modeling](https://mira-multiome.readthedocs.io/en/latest/notebooks/tutorial_topic_analysis.html)
- scRNAseq data QC preprocessing and topic modeling
  
Step 2: [MIRA Tutorial - ATAC Topic Modeling](https://mira-multiome.readthedocs.io/en/latest/notebooks/tutorial_atlas_integration.html)
- scATACseq data QC preprocessing and topic modeling
  
Step 3: [MIRA Tutorial - Joint Representation](https://mira-multiome.readthedocs.io/en/latest/notebooks/tutorial_joint_representation.html)
- Characterize cells by the expression and accessibility topics
  
Step 4: [MIRA Tutorial - Topic Analysis](https://mira-multiome.readthedocs.io/en/latest/notebooks/tutorial_topic_analysis.html)
- Analyze the topics to understand the regulatory dynamics present in a sample
  
Step 5: [MIRA Tutorial -  Regulatory Potential Modeling](https://mira-multiome.readthedocs.io/en/latest/notebooks/tutorial_cisregulatory_modeling.html)
- Creates a LITE model, which learns the exponential decay function settings to use for each target gene to find CREs
  
Step 6: [MIRA Tutorial - NITE Regulation](https://mira-multiome.readthedocs.io/en/latest/notebooks/tutorial_NITE_LITE_modeling.html)
- Extends the LITE model to relate genome-wide changes in chromatin to gene expression via the learned accessibility topics


## Step 1: RNA Topic Modeling
#### Feature Selection and Preprocessing
Filter out genes expression in fewer than 15 cells

```python
    sc.pp.filter_genes(data, min_cells=15)
```
Log1p CPM normlize the data

```python
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
```
Only keep variable genes (expression dispersion > 0.5)

```python
    sc.pp.highly_variable_genes(data, min_disp = 0.5)
```
Save a version of the raw counts to `data.layers['counts']`

#### Creating the Expression Topic Model
Instantiate a MIRA ExpressionModel object

```python
    model = mira.topics.make_model(
        data.n_obs, data.n_vars, # helps MIRA choose reasonable values for some hyperparameters which are not tuned.
        feature_type = 'expression',
        highly_variable_key='highly_variable',
        counts_layer='counts',
    )
```
Find the minimum and maximum bounds for the learning rate
```python
    min_lr, max_lr = model.get_learning_rate_bounds(adata)
    model.set_learning_rates(min_lr, max_lr)
```

#### Hyperparameter Tuning
**Method 1: Gradient-based**
Find the optimal number of topics to use
```python
    topic_contributions = mira.topics.gradient_tune(model, adata)
    sig_topic_contributions = [x for x in topic_contributions if x > 0.05]
    
    # Contribution of each topic plateaus, find the elbow to set threshold
    kneedle = KneeLocator(
    np.arange(len(log_contributions)), 
    log_contributions,
    curve="convex",
    direction="decreasing",
    online=False
    )
    num_topics = math.ceil(kneedle.elbow or 2)
```

**Method 2: Bayesian Optimization (optional, but much more comprehensive)**
Create and train a BayesianTuner, setting the `min_topics` and `max_topics` based on the results from method 1

```python
    tuner = mira.topics.BayesianTuner(
    model = model,
    n_jobs=n_jobs,
    save_name = tuner_save_name,
    min_topics = max(1, num_topics - 5), 
    max_topics = min(50, num_topics + 5),
    )
    tuner.fit(adata)
```

Fetch the best weights from the trained tuner

```python
    model = tuner.fetch_best_weights()
```

Save the model

```
    model.save(model_save_path)
```

## Step 2: ATAC Topic Modeling
#### Preprocessing
Filter out genes expression in fewer than 30 cells

```python
    sc.pp.filter_genes(data, min_cells = 30)
```
Filter out cells that are not also in the pre-processed RNA dataset

```python
    rna_adata = anndata.read_h5ad(rna_h5ad_save_path)
    barcodes = rna_adata.obs_names.to_list()
    atac_adata = atac_adata[barcodes]
```
Filter out cells with fewer than 1000 peaks per cell

```python
    sc.pp.filter_cells(data, min_genes=1000)
```
Downsample to 100K peaks (if there are more than 100K peaks)

```python
    np.random.seed(0)
    atac_adata.var['endogenous_peaks'] = np.random.rand(atac_adata.shape[1]) <= min(1e5/atac_adata.shape[1], 1)
```
#### Creating the Accessibility Topic Model
Instantiate a MIRA AccessibilityModel object

```python
    model: AccessibilityModel = mira.topics.make_model(
    *atac_adata.shape,
    feature_type = 'accessibility',
    endogenous_key='endogenous_peaks', # which peaks are used by the encoder network
    atac_encoder="DAN"
    )
```
Cache the train/test split to the disk

```python
    train_dir = os.path.join(training_cache, 'atac_train')
    test_dir = os.path.join(training_cache, 'atac_test')
    
    if not os.path.exists(train_dir):
    model.write_ondisk_dataset(train, dirname=train_dir)
    if not os.path.exists(test_dir):
        model.write_ondisk_dataset(test, dirname=test_dir)
```
Find the minimum and maximum bounds for the learning rate

```python
    min_lr, max_lr = model.get_learning_rate_bounds(adata)
    model.set_learning_rates(min_lr, max_lr)
```
#### Hyperparameter Tuning
**Method : Gradient-based**
Find the optimal number of topics to use

    ```python
        topic_contributions = mira.topics.gradient_tune(model, adata)
        sig_topic_contributions = [x for x in topic_contributions if x > 0.05]
        
        # Contribution of each topic plateaus, find the elbow to set threshold
        kneedle = KneeLocator(
        np.arange(len(log_contributions)), 
        log_contributions,
        curve="convex",
        direction="decreasing",
        online=False
        )
        num_topics = math.ceil(kneedle.elbow or 2)
    ```
**Method 2: Bayesian Optimization (optional, but much more comprehensive)**
Create and train a BayesianTuner, setting the `min_topics` and `max_topics` based on the results from method 1

    ```python
        tuner = mira.topics.BayesianTuner(
        model = model,
        n_jobs=n_jobs,
        save_name = tuner_save_name,
        min_topics = max(1, num_topics - 5), 
        max_topics = min(50, num_topics + 5),
        )
        tuner.fit(adata)
    ```
Fetch the best weights from the trained tuner

    ```python
        model = tuner.fetch_best_weights()
    ```
Save the model

```python
model.save(model_save_path)
```
## Step 3: Joint Representation
The topic models are joined to represent different states of chromatin accessibility and gene expression. We can then represent a cell's state based on how much each topic is represented by its gene expression.
#### Load Topic Models and Processed Data
Load the processed RNA and ATAC AnnData objects

```python
    rna_adata = anndata.read_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/rna_data.h5ad")
    atac_adata = anndata.read_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/atac_data.h5ad")
```
Load the trained topic models

```python
    rna_model = mira.topics.load_model("/gpfs/Home/esm5360/MIRA/mira-datasets/rna_model.pth")
    atac_model = mira.topics.load_model("/gpfs/Home/esm5360/MIRA/mira-datasets/atac_model.pth")
```
#### Predict Topic Compositions for the Cells
Save the topic compositions for cells and features to the AnnData objects

```python
    atac_model.predict(atac_adata)
    rna_model.predict(rna_adata)
```
Find cells with similar topic compositions

```python
    rna_model.get_umap_features(rna_adata, box_cox='log')
    atac_model.get_umap_features(atac_adata, box_cox='log')
```
#### Joining Modalities
Construct joint embedding of the expression topics and accessibility topics for each cell

```python
    rna_adata, atac_adata = mira.utils.make_joint_representation(rna_adata, atac_adata)
```
Transfer metadata from the ATAC DataFrame to the RNA dataframe so that there is one main object for plotting and running other functions

```python
    rna_adata.obs = rna_adata.obs.join(
        atac_adata.obs.add_prefix('ATAC_')
    )
    
    atac_adata.obsm['X_umap'] = rna_adata.obsm['X_umap']
```
#### Analyzing Joint Topic Compositions
Determine the degree to which one modality's topics correspond with topics in the other modality using pointwise mutual information

```python
    mira.tl.get_cell_pointwise_mutual_information(rna_adata, atac_adata)
```
Summarize the mutual information across all cells using concordance (0 - low concordance, 0.5 - high concordance)

```python
    mutual_info_score = mira.tl.summarize_mutual_information(rna_adata, atac_adata)
    print("Mutual information score (0 - low concordence, 0.5 - high concordance)")
    print(mutual_info_score)
```
Find the cross-correlation between the ATAC and RNA topics to see which ones are related

```python
    cross_correlation = mira.tl.get_topic_cross_correlation(rna_adata, atac_adata)
```
Save the AnnData objects

```python
atac_adata.write_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/atac_data_joint_representation.h5ad")
rna_adata.write_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/rna_data_joint_representation.h5ad")
```
## Step 4: Topic Analysis
The trained topic models and joint-KNN representation of the data can be used to understand the regulatory dynamics.
**Expression topics**: Functional enrichment of top genes activated in a topic / module.
**Accessibility topics**: Find transcription factor regulators of particular cell states.
#### Load Topic Models and Joint Representation Data from Step 3
Load the processed RNA and ATAC AnnData objects from the joint representation step

```python
    rna_adata = anndata.read_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/rna_data_joint_representation.h5ad")
    atac_adata = anndata.read_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/atac_data_joint_representation.h5ad")
```
Load the trained topic models

```python
    rna_model = mira.topics.load_model("/gpfs/Home/esm5360/MIRA/mira-datasets/rna_model.pth")
    atac_model = mira.topics.load_model("/gpfs/Home/esm5360/MIRA/mira-datasets/atac_model.pth")
```
#### Expression Topic Analysis
MIRA uses Enrichr to get functional enrichments for each topic. Post the top 5% of genes modeled by the expression topic model

```python
    num_genes = rna_adata.X.shape[0]
    top_n_genes = math.ceil(num_genes * 0.05)
    
    rna_model.post_topics(top_n=top_n_genes)
```
Download the enrichment results for all topics and compare GO terms against WikiPathways

```python
    rna_model.fetch_enrichments(ontologies=['WikiPathways_2019_Mouse'])
```
#### Accessibility Topic Analysis
Download the mm10 genome fasta sequence from goldenPath (or use existing path)

```bash
    mkdir -p data
    wget https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz -O /gpfs/Home/esm5360/MIRA/data/mm10.fa.gz
    cd data/ && gzip -d -f mm10.fa.gz
```
Ensure that `atac_adata.var` contains the `peak_id`, `chr`, `start`, and `end` columns

```python
    peak_locations = atac_adata.var.index
    
    if not any(["chr", "start", "end"]) in peak_locations:
        peak_data = {
            "peak_id": [],
            "chr": [],
            "start": [],
            "end": []
        }
        for i, peak in enumerate(peak_locations):
            peak_id = i
            chr_num = peak.split(":")[0]
            peak_start = int(peak.split(":")[1].split("-")[0])
            peak_end = int(peak.split(":")[1].split("-")[1])
            
            peak_data["peak_id"].append(peak_id)
            peak_data["chr"].append(chr_num)
            peak_data["start"].append(peak_start)
            peak_data["end"].append(peak_end)
            
        peak_df = pd.DataFrame(peak_data, index=peak_locations)
        atac_adata.var = pd.concat([atac_adata.var, peak_df], axis=1)
```
Make sure that `moods-dna.py` is in the `PATH` variable

```python
    os.environ["PATH"] = os.pathsep.join([
        os.path.expanduser("~/miniconda3/envs/mira-env/bin"),
        os.environ["PATH"]
    ])
```
Run MIRA motif scanning against the JASPAR 2020 vertebrates collection of motifs

```python
    mira.tools.motif_scan.logger.setLevel(logging.INFO) # make sure progress messages are displayed
    mira.tl.get_motif_hits_in_peaks(atac_adata,
                        genome_fasta='/gpfs/Home/esm5360/MIRA/data/mm10.fa',
                        chrom = 'chr', start = 'start', end = 'end',
                        pvalue_threshold=1e-4
                        ) # indicate chrom, start, end of peaks
```
Filter out TFs that dont have any associated data in the RNA AnnData object (gene names from this part are uppercase, be careful! Our mouse data is capitalized so it needs to be converted)

```python
    mira.utils.subset_factors(atac_adata,
                            use_factors=[factor.upper() for factor in rna_adata.var_names
                                            if not ('FOS' in factor or 'JUN' in factor)])
```
Get the enriched TFs for each topic

```python
    topics = [int(i.replace("topic_", "")) for i in atac_adata.obs if "topic" in i]
    for topic in topics:
        atac_model.get_enriched_TFs(atac_adata, topic_num=topic, top_quantile=0.1)
```
#### Motif Scoring
MIRA calculates motif scores as the log-probability of sampling a motif-associated region from the posterior predictive distribution over regions from the topic model.

```python
    motif_scores = atac_model.get_motif_scores(atac_adata)
    
    # Reformat to make it convenient for plotting
    motif_scores.var = motif_scores.var.set_index('parsed_name')
    motif_scores.var_names_make_unique()
    motif_scores.obsm['X_umap'] = atac_data.obsm['X_umap']
```
Save the AnnData objects

```python
rna_adata.write_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/rna_data_topic_analysis.h5ad")
atac_adata.write_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/atac_data_topic_analysis.h5ad")
```
## Step 5: Regulatory Potential Modeling - LITE Regulation
MIRA learns upstream and downstream TSS to CRE decay distances for scoring the regulatory influence a CRE has over a gene. MIRA learns the distances that describe the range at which changes in local accessible chromatin appear to influence expression. Each RP model is a statistical model describing a gene's relationship with it's local chromatin neighborhood. 

This model can be used to associate TF-TG by measuring the ability of the RP model to predict the expression of a gene before and after the REs predicted to bind a certain TF are masked. To get driver TFs which regulate the differences between topics that make those topics unique, MIRA uses a Wilcoxon test to compare how well TF expression is associated across many co-regulated genes in a topic. This is useful for finding TFs that control the differences between cell states.  
#### Load Topic Models and Topic Analysis Data from Step 4
Load the processed RNA and ATAC AnnData objects from the topic analysis step

```python
    rna_adata = anndata.read_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/rna_data_topic_analysis.h5ad")
    atac_adata = anndata.read_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/atac_data_topic_analysis.h5ad")
```
Load the trained topic models

```python
    ds011_atac_model = mira.topics.load_model("/gpfs/Home/esm5360/MIRA/mira-datasets/atac_model.pth")
    ds011_rna_model = mira.topics.load_model("/gpfs/Home/esm5360/MIRA/mira-datasets/rna_model.pth")
```
#### TSS Annotations
Download the TSS annotations and chrom sizes from MIRA (or specify our own, but the format is a bit different)

```python
    mira.datasets.mm10_chrom_sizes()
    mira.datasets.mm10_tss_data()
```
Ensure that `atac_adata.var` contains the `peak_id`, `chr`, `start`, and `end` columns

```python
    peak_locations = atac_adata.var.index
    
    if not any(["chr", "start", "end"]) in peak_locations:
        peak_data = {
            "peak_id": [],
            "chr": [],
            "start": [],
            "end": []
        }
        for i, peak in enumerate(peak_locations):
            peak_id = i
            chr_num = peak.split(":")[0]
            peak_start = int(peak.split(":")[1].split("-")[0])
            peak_end = int(peak.split(":")[1].split("-")[1])
            
            peak_data["peak_id"].append(peak_id)
            peak_data["chr"].append(chr_num)
            peak_data["start"].append(peak_start)
            peak_data["end"].append(peak_end)
            
        peak_df = pd.DataFrame(peak_data, index=peak_locations)
        atac_adata.var = pd.concat([atac_adata.var, peak_df], axis=1)
```
Read in and capitalize the TSS gene names to match the gene names in the dataset

```python
    tss_data_file = 'mira-datasets/mm10_tss_data.bed12'
    tss_data = pd.read_csv('mira-datasets/mm10_tss_data.bed12', sep="\t")
    tss_data["#geneSymbol"] = tss_data["#geneSymbol"].str.capitalize()
    tss_data.to_csv('mira-datasets/mm10_tss_data.bed12', sep="\t", header=True, index=False)
```
Calculate the distance between the peaks and gene TSSs

```python
    mira.tl.get_distance_to_TSS(atac_adata,
                                tss_data='mira-datasets/mm10_tss_data.bed12',
                                genome_file='mira-datasets/mm10.chrom.sizes')
```
Resave the ATAC AnnData object after calculating the TSS distance

```python
    atac_adata.write_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/ds011_atac_data_tss_dist.h5ad")
```
#### Regulatory Potential Model Training
Determine the genes for training

All highly-variable genes plus all genes that scored in the top 5% most-activated for any topic
```python
    # Ensure all gene names are uppercased to match TSS annotation
    rp_genes = list(ds011_rna_model.features[ds011_rna_model.highly_variable])
    for topic in range(ds011_rna_model.num_topics):
        rp_genes.extend(ds011_rna_model.get_top_genes(topic, 200))
    
    # Capitalize and deduplicate
    rp_genes = list(set(g.capitalize() for g in rp_genes if g.capitalize() in rna_adata.var_names))
```
Instantiate an RP LITE model

```python
    litemodel = mira.rp.LITE_Model(expr_model = ds011_rna_model,
                                accessibility_model=ds011_atac_model,
                                genes = rp_genes)
```
Set the rna_adata.X value to the raw counts (otherwise it doesn't work) using the "counts" layer created when running preprocessing

```python
    rna_adata.X = rna_adata.layers["counts"]
```
Map the AnnData objects to an `rp_args` dictionary for `expr_adata` and `atac_adata` to make it convenient to put them into RP-model related functions

```python
    rp_args = dict(expr_adata = rna_adata, atac_adata= atac_adata)
```
Train the LITE model using the RNA topic model and ATAC topic model

```python
    litemodel.fit(
        **rp_args,
        n_workers=4,
        callback=mira.rp.SaveCallback('/gpfs/Home/esm5360/MIRA/data/ds011_rpmodels/')
    )
```
#### Defining Local Chromatin Neighborhoods
Create a DataFrame of the parameters for each gene's regulatory potential decay distance (distance is decay rate in kilobases)

```python
    TSS_dist_decay_df = pd.DataFrame(
        litemodel.parameters_
    ).T
```
Get all peaks within the influence of a gene's RP model

```python
    litemodel['Gpc6'].get_influential_local_peaks(atac_adata, decay_periods = 5.).head(5)
```
**Compute MIRA regulatory potential scores using the distance to TSS and LITE model gene-specific decay rates**

```python
    def compute_rp_score(row, params):
        dist_kb = row['distance_to_TSS'] / 1000  # convert bp → kb
        if row['is_upstream']:
            decay = params['distance_upstream']
            weight = params['a_upstream']
        else:
            decay = params['distance_downstream']
            weight = params['a_downstream']
        return weight * np.exp(-dist_kb / decay)
    
    all_rp_records = []
    
    missing_genes = 0
    
    for gene in rna_adata.var_names:
        try:
            params = litemodel[gene].parameters_
            peaks_df = litemodel[gene].get_influential_local_peaks(atac_adata, decay_periods=5.)
    
            peaks_df['MIRA_LITE_RP_score'] = peaks_df.apply(lambda row: compute_rp_score(row, params), axis=1)
    
            df = peaks_df[["distance_to_TSS", "MIRA_LITE_RP_score"]].rename_axis("peak_id").reset_index()
            df["target_id"] = gene
            df = df[["peak_id", "target_id", "distance_to_TSS", "MIRA_LITE_RP_score"]]
    
            all_rp_records.append(df)
    
        except IndexError:
            missing_genes += 1
        except KeyError:
            print(f"Gene {gene} not found in litemodel — skipping.")
        except Exception as e:
            print(f"Error processing {gene}: {e}")
    
    print(f"Missing models for {missing_genes}")
    
    regulatory_potential_df = pd.concat(all_rp_records, ignore_index=True)
```
#### Predicting Expression from Accessibility
Use the trained RP model to calculate the maximum aposteriori prediction of expression given the accessibility state of each gene in each cell and quantify the likelihood of that prediction

```python
    litemodel.predict(**rp_args)
```
#### Gene-TF Targeting
Use probabilistic *in-silico* deletion (pISD) to find potential regulatory associations between genes and TFs.

Measures the ability of the RP model to predict the expression of a gene before and after the regulatory elements predicted to bind a certain transcription factor are masked
```python
    litemodel.probabilistic_isd(**rp_args, n_workers = 4)
```
Fetch the matrix of association scores between gene-TF pairs

```python
    isd_matrix = mira.utils.fetch_ISD_matrix(rna_data)
```
Save the RNA AnnData object with the matrix of TF-TG trans-regulatory potential scores

```python
    rna_adata.write_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/rna_data_tf_tg_scores.h5ad")
```
rna_adata.write_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/rna_data_tf_tg_scores.h5ad")
#### Querying with many genes
Find potential driver TFs which regulate many genes in a topic

```python
    isd_results = mira.tl.driver_TF_test(rna_adata, geneset=ds011_rna_model.get_top_genes(0, 150))
```
Plot differential driver TF enrichment between topics

```python
    mira.pl.compare_driver_TFs_plot(rna_adata,
                                    geneset1=ds011_rna_model.get_top_genes(0, 150),
                                    geneset2=ds011_rna_model.get_top_genes(7, 150),
                                    fontsize=20, figsize=(5,5), color='lightgrey',
                                    axlabels= ('-log10 p-value\Topic 0 Regulators','-log10 p-value\Topic 7 Regulators'))
    plt.show()
```
#### Visualizing RP Models
Find a TG of interest by max TF association score

```python
    isd_matrix.max(axis=1).sort_values(ascending=False)
```
Write the RP model profile for the TG of interest to a bedfile

```python
    litemodel['Cxxc4'].write_bedgraph(atac_adata,
                                    save_name = '/gpfs/Home/esm5360/MIRA/data/trackdata/Cxxc4_rpmodel.bedgraph')
```
Identify a TF with a high association score with the TG of interest

```python
    isd_matrix.loc['Cxxc4'].sort_values().tail(10)
```
Identify binding sites for the TF and save as a hits bed file

```python
    # Get the unique ID for the TF motif
    factors = mira.utils.fetch_factor_meta(atac_adata)
    factors[factors.name == 'EGR1']
    
    # Fetch the motif binding sites
    egr1_binding = mira.utils.fetch_binding_sites(atac_adata, id = 'MA0162.4')
    
    # Sort by chromosome and start site, then save to a TF hits.bed file
    egr1_binding[['chr','start','end']].sort_values(['chr','start'])\
        .to_csv('data/trackdata/Egr1_hits.bed',  # save as bed file
                index = None, header=None, sep = '\t',
                )
```
Create a pygenometracks configuration file

```python
    config_file = """
    
    [x-axis]
    
    [genes]
    file = mira-datasets/mm10_tss_data.bed12
    title = Genes
    height = 3
    
    [rp model]
    file = data/trackdata/Cxxc4_rpmodel.bedgraph
    height = 3
    color = #e6e6e6
    title = CXXC4 RP Model
    max_value = 1.1
    min_value = 0
    file_type = bedgraph
    alpha = 0.2
    
    [rp model2]
    file = data/trackdata/Cxxc4_rpmodel.bedgraph
    type = line:0.5
    color = black
    file_type = bedgraph
    max_value = 1.1
    min_value = 0
    overlay previous = yes
    
    [spacer]
    
    [Egr1 hits]
    file = data/trackdata/Egr1_hits.bed
    height = 1.5
    style = UCSC
    gene_rows = 1
    color = red
    title = EGR1 Motif Hits
    labels = off
    
    """
    
    with open('data/trackdata/config.ini', 'w') as f:
        print(config_file, file = f)
```
Find the TSS for the TG of interest to set the view for the plot

```python
    mira.utils.fetch_TSS_data(atac_adata)['Cxxc4']
```
Plot the TF binding sites for the TG of interest and the TG's distance decay rate

```python
    !/gpfs/Home/esm5360/miniconda3/envs/mira-env/bin/pyGenomeTracks \
        --tracks data/trackdata/config.ini \
        --region chr3:134100000-134400000 \
        -out data/trackdata/Cxxc4_plot.png \
        --dpi 300 \
        --width 20 \
        --fontSize 6
```

## Step 6: NITE Regulation
The authors of MIRA found that local chromatin accessibility does not always correspond to changes in gene expression. For genes that aren't regulated by local interactions, the LITE model doesn't work. The NITE model extends the LITE model by looking at how a gene's expression changes due to genome-wide chromatin changes. As we have topic models of the chromatin accessibility, we have a summary of the different accessibility states.

**Chromatin Differential Score**: Calculates the disagreement between the NITE and LITE model,  increases as the LITE model over-estimates expression (higher chromatin differential = higher LITE model error).  

**NITE Scores**: One-number summary metric describing the statistical divergence between gene expression and local chromatin accessibility  

NITE scores essentially act as a beacon for interesting regulation, as divergence of local chromatin and gene expression appears to be influenced by important processes such as signaling, lineage priming, and fate commitment.  
#### Load Topic Models and Topic Analysis Data from Step 4
Load the processed RNA and ATAC AnnData objects from the topic analysis step

```python
    rna_adata = anndata.read_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/ds011_rna_data_tf_tg_scores.h5ad")
    atac_adata = anndata.read_h5ad("/gpfs/Home/esm5360/MIRA/mira-datasets/ds011_atac_data_tss_dist.h5ad")
```
Load the trained topic models

```python
    ds011_atac_model = mira.topics.load_model("/gpfs/Home/esm5360/MIRA/mira-datasets/atac_model.pth")
    ds011_rna_model = mira.topics.load_model("/gpfs/Home/esm5360/MIRA/mira-datasets/rna_model.pth")
```
#### Training NITE Models
Load in the LITE model and run the predictions on the AnnData objects

```python
    rp_args = dict(expr_adata = rna_adata, atac_adata = atac_adata)
    
    litemodel = mira.rp.LITE_Model.load_dir(
        expr_model = ds011_rna_model,
        accessibility_model = ds011_atac_model,
        prefix='/gpfs/Home/esm5360/MIRA/data/ds011_rpmodels/'
    )
    litemodel.predict(**rp_args)
```
Initialize the NITE model

```python
    nitemodel = litemodel.spawn_NITE_model()
```
Fit the NITE model parameters and predict gene expression given genome-wide chromatin state

```python
    nitemodel.fit(**rp_args, n_workers=4)
    nitemodel.predict(**rp_args)
```
> NOTE: I had to add a line with `NITE_features = NITE_features.float()` at line 927 of `mira/rp_model.py` for `nitemodel.predict()` to work, or else I got an error `RuntimeError: expected scalar type Double but found Float`  
Save the NITE model

```python
    nitemodel.save("/gpfs/Home/esm5360/MIRA/data/ds011_rpmodels/")
```
#### Chromatin Differential
Calculate the chromatin differential between the LITE model and the NITE model

```python
    mira.tl.get_chromatin_differential(rna_adata)
```
Create a DataFrame of the cell x gene chromatin differentials

```python
    chrom_diff = pd.DataFrame(rna_adata.layers["chromatin_differential"].toarray(), columns=rna_adata.var_names, index=rna_adata.obs_names)
```
Plot the chromatin differential figures to assess genes where the models don't agree

```python
    mira.pl.plot_chromatin_differential(rna_adata, genes = litemodel.genes,
                                        show_legend=False, size = 0.1, aspect=1.2)
    plt.show()
```
**Add the mean gene chromatin differential scores to the peak-to-TG regulatory potential DataFrame**

```python
    # Get the mean chromatin differential scores for each gene
    gene_cell_chromatin_diff = pd.DataFrame(rna_adata.layers["chromatin_differential"].toarray().T, columns=rna_adata.obs_names, index=rna_adata.var_names)
    avg_chrom_diff = gene_cell_chromatin_diff.mean(axis=1)
    
    # Add the mean chromatin differential scores to the RP score DataFrame
    regulatory_potential_df["avg_chromatin_differential"] = regulatory_potential_df["target_id"].map(avg_chrom_diff)
    regulatory_potential_df.to_csv("/gpfs/Home/esm5360/MIRA/mira-datasets/ds011_peak_to_gene_lite_rp_score_chrom_diff.csv")
```
#### NITE Scores
Get the NITE scores for genes and cell states, describes statistical divergence between gene expression and local chromatin accessibility

```python
    mira.tl.get_NITE_score_cells(rna_adata)
    mira.tl.get_NITE_score_genes(rna_adata)
```
Note: to conduct NITE score testing with < 2000 genes, manually calculate and provide a measure of the median non-zero count rate across genes

    ```python
        median_nonzero_counts = np.median(
                np.array((rna_adata[:, rna_model.features].X > 0).sum(-1))
            ) # get median number of nonzero counts across genes
        
        mira.tl.get_NITE_score_cells(rna_adata, median_nonzero_expression=median_nonzero_counts)
        mira.tl.get_NITE_score_genes(rna_adata, median_nonzero_expression=median_nonzero_counts)
    ```
#### Cell-level NITE Score
Plot NITE scores for cells, highlights cell states where local chromatin accessibility is less predictive of gene expression

```python
    fig, ax = plt.subplots(1,1,figsize=(15,10))
    sc.pl.umap(rna_adata, color = 'NITE_score', vmin = 0, vmax = 'p99', color_map = 'BuPu',
                    frameon=False, ax= ax, size = 15)
```
#### Gene-level NITE Score
Plot NITE scores for genes, orders genes according to how predictive their local chromatin environment is of gene expression expression

```python
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax = sns.kdeplot(data = rna_data.var, x = 'NITE_score', fill = True,
                    color = 'lightgrey', edgecolor = 'black', cut = 0 )
    for i, gene in enumerate(litemodel.genes):
        nitescore = rna_data.var.loc[gene].NITE_score
        ax.vlines(nitescore, ymin = 0, ymax = 0.005, color = 'black')
        ax.text(x = nitescore, y = 0.006, s = gene)
    
    ax.set(xlim = (0, 100))
    sns.despine()
```
