# HowTo Run experiments

## Generating the require meta-data

*Note:* Choose your setting. All following calls will **rely** on this variable being set 
correctly. For the paper we employed the following settings:
```
EXP=10MIN
EXP=60MIN
```

*Note:* each call to `master_calls.py` will create one or more new commands file. Execute all 
commands in that file before moving to the next stage.

### 1. ASKL on metadata (TR1)

Generate metadata to later on create a configuration-task performance matrix which will be used
to construct the portfolios 

```
python master_calls.py --do ASKL_metadata --setting $EXP
```

### 2. Create Matrix (TR2)

```
python master_calls.py --do ASKL_getportfolio --setting $EXP
```
*Note:* It can be that not all entries of the matrix are computed within the given runtime with 
the amount of workers started by the script, so you just have to run the commands in the 
commands files again.

```
python master_calls.py --do run_create_matrix --setting $EXP
python master_calls.py --do run_create_symlinks --setting $EXP
```

### 3. Create Portfolio (TR3)
```
python master_calls.py --do ASKL_create_portfolio --setting $EXP
```
*Note*: This produces two kinds of commands files: ones that are called `build_portfolio.cmd` 
and `build_portfolio_cv.cmd` and other that start with `execute_portfolio`. Execute the `build` 
command files first and only then execute the latter ones.

### 4. Create Metadata (TR4)

```
python master_calls.py --do ASKL_metadata_full_run_with_portfolio  --setting $EXP
```

### 5. Build Selector and create meta-data (TR5&6)
```
python master_calls.py --do RQ1_AutoAuto_build --setting %s 
```

### 6. Obtain results (TE1&2)

Here we first run all policies on all metadatasets and afterwards only create symlinks where 
necessary. By this, the policy data can be used to both assess the quality of the individual 
policies and also the selector.

```
python master_calls.py --do ASKL_automldata_run_with_portfolio_w_ensemble  --setting $EXP
python master_calls.py --do RQ1_AutoAuto_simulate --setting %s 
```

### 7. Run baselines

```
python master_calls.py --do ASKL_automldata_w_ensemble --setting $EXP
python master_calls.py --do ASKL_automldata_w_ensemble_w_knd --setting $EXP
```

### 8. Ablation studies

```
python3 askl2paper_do.py --version askl2papertest --mode $EXP --do RQ2.3_AutoAuto_build
python3 askl2paper_do.py --version askl2papertest --mode $EXP --do RQ2.3_AutoAuto_simulate

python3 askl2paper_do.py --version askl2papertest --mode $EXP --do RQ2.1_AutoAuto_build
python3 askl2paper_do.py --version askl2papertest --mode $EXP --do RQ2.1_AutoAuto_simulate

python3 askl2paper_do.py --version askl2papertest --mode 1MIN --do ASKL_metadata_full
python3 askl2paper_do.py --version askl2papertest --mode 1MIN --do RQ2.2_AutoAuto_build
python3 askl2paper_do.py --version askl2papertest --mode 1MIN --do RQ2.2_AutoAuto_simulate
```
