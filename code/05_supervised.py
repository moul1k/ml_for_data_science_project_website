"""Module 3: fixed-snapshot discovery-method classification.

Run from any directory: python code/05_supervised.py
No downloads. Feature transformations and selection are fitted on training folds.
The final test split is fixed before any tuning; no model is selected on test scores.
"""
from pathlib import Path
import hashlib
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/module3"
FIG = ROOT / "figures/module3"
REPORT = ROOT / "reports/module3"
FEATURES = ["pl_rade", "pl_bmasse", "pl_orbper", "st_teff", "st_mass", "st_rad"]
NAMES = ["Planet radius", "Planet mass", "Orbital period", "Star temperature", "Star mass", "Star radius"]
UNITS = ["Earth radii", "Earth masses", "days", "K", "solar masses", "solar radii"]
CLASSES = ["Radial Velocity", "Transit"]
COLORS = ["#ee7733", "#4477aa"]
SEED = 42


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_json(obj, name):
    (REPORT / name).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def preview(frame, name, title):
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")
    display = frame.head(5).copy()
    for c in display.select_dtypes("number"): display[c] = display[c].round(3)
    table = ax.table(cellText=display.astype(str).values, colLabels=display.columns,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.9)
    ax.set_title(title, pad=20); save(fig, name)


def overview():
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for xy, text in [((.5,.87), "Period <= threshold?"), ((.24,.47), "Predict Transit"),
                     ((.76,.47), "Mass <= threshold?"), ((.6,.1), "Predict Transit"),
                     ((.9,.1), "Predict Radial Velocity")]:
        ax.text(*xy, text, ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round,pad=.6", fc="#edf2f7", ec="#4477aa"))
    for start,end,label in [((.5,.79),(.24,.56),"yes"),((.5,.79),(.76,.56),"no"),
                             ((.76,.39),(.6,.19),"yes"),((.76,.39),(.9,.19),"no")]:
        ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",color="#475569"))
        ax.text((start[0]+end[0])/2,(start[1]+end[1])/2,label,fontsize=10)
    ax.set(xlim=(0,1.12),ylim=(-.03,1.02), title="Recursive classification | illustrative rules, not a fitted result")
    ax.axis("off"); save(fig,"tree_concept")
    p=np.linspace(.0001,.9999,500)
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(p,-p*np.log2(p)-(1-p)*np.log2(1-p),label="Entropy (bits)",color=COLORS[0])
    ax.plot(p,2*p*(1-p),label="Gini impurity",color=COLORS[1])
    ax.set(xlabel="Proportion in one class",ylabel="Node impurity (different scales)",
           title="Pure nodes have zero impurity | theoretical curves",xlim=(0,1),ylim=(0,1.05))
    ax.legend(); ax.grid(alpha=.15); save(fig,"tree_impurity")


def metrics(model, x, y):
    pred=model.predict(x)
    return {"accuracy":float(accuracy_score(y,pred)),
            "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "confusion_matrix":confusion_matrix(y,pred,labels=CLASSES).tolist(),
            "classification_report":classification_report(y,pred,labels=CLASSES,output_dict=True,zero_division=0)}


def confusion_plot(ax, cm, title):
    cm=np.asarray(cm)
    ax.imshow(cm,cmap="Blues",vmin=0,vmax=max(cm.max(),1))
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=14,
                                  color="white" if cm[i,j]>cm.max()*.55 else "#17243d")
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Radial velocity","Transit"],
           yticklabels=["Radial velocity","Transit"],xlabel="Predicted method",ylabel="Recorded method",title=title)


def bootstrap_interval(y, pred, groups):
    """Conditional holdout uncertainty; resample group-proxy blocks, not rows."""
    unique=np.unique(groups)
    matrices=np.array([confusion_matrix(y[groups==g],pred[groups==g],labels=CLASSES) for g in unique])
    rng=np.random.default_rng(SEED)
    values=[]
    for _ in range(1000):
        cm=matrices[rng.integers(0,len(unique),len(unique))].sum(axis=0)
        if np.any(cm.sum(axis=1)==0):continue
        values.append([np.trace(cm)/cm.sum(), np.mean(np.diag(cm)/cm.sum(axis=1))])
    return {"accuracy_95pct":np.quantile(values,[.025,.975],axis=0)[:,0].tolist(),
            "balanced_accuracy_95pct":np.quantile(values,[.025,.975],axis=0)[:,1].tolist(),
            "replicates":len(values),"unit":"planet-name host proxy; fixed fitted model"}


def run():
    for folder in [DATA,FIG,REPORT]:folder.mkdir(parents=True,exist_ok=True)
    source=ROOT/"data/processed/exoplanets_clean.csv"
    raw=pd.read_csv(source)
    is_target=raw.discoverymethod.isin(CLASSES)
    valid=np.isfinite(raw[FEATURES]).all(axis=1)&(raw[FEATURES]>0).all(axis=1)
    frame=raw.loc[is_target&valid].copy()
    frame.insert(0,"source_row",frame.index)
    frame=frame.sort_values("pl_name").reset_index(drop=True)
    # No verified hostname in the supplied snapshot. Keep conventional b/c/d
    # siblings together using a transparent, explicitly imperfect name proxy.
    frame["host_proxy"]=frame.pl_name.str.replace(r"\s+[b-z]$","",regex=True)
    x=frame[FEATURES]; y=frame.discoverymethod; groups=frame.host_proxy
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    train,test=next(splitter.split(x,y,groups))
    assert not set(train)&set(test)
    assert not set(groups.iloc[train])&set(groups.iloc[test])
    assert not set(frame.pl_name.iloc[train])&set(frame.pl_name.iloc[test])
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=43).split(x.iloc[train],y.iloc[train],groups.iloc[train]))
    for a,b in folds:assert not set(groups.iloc[train].iloc[a])&set(groups.iloc[train].iloc[b])
    overview()
    metadata=frame[["source_row","pl_name","host_proxy","discoverymethod"]].copy()
    metadata["split"]="train"; metadata.loc[test,"split"]="test"
    metadata.to_csv(DATA/"split_manifest.csv",index=False)
    cv_manifest=metadata.iloc[train].copy()
    cv_manifest["validation_fold"]=-1
    for fold,(_,b) in enumerate(folds,1):cv_manifest.iloc[b,cv_manifest.columns.get_loc("validation_fold")]=fold
    cv_manifest.to_csv(DATA/"training_cv_folds.csv",index=False)
    for label,idx in [("labeled",np.arange(len(frame))),("train",train),("test",test)]:
        table=frame.loc[idx,FEATURES+["discoverymethod"]]
        table.to_csv(DATA/f"{label}.csv",index=False)
        table.head(12).to_csv(DATA/f"{label}_sample.csv",index=False)
        display=table.rename(columns=dict(zip(FEATURES,["R (Earth)","M (Earth)","P (days)","T star (K)","M star (Sun)","R star (Sun)"])))
        preview(display,f"{label}_preview",f"{label.capitalize()} physical data | first five rows; target is discoverymethod")
    retention=pd.crosstab(raw.discoverymethod,np.where(valid,"Eligible measurements","Excluded measurements"))
    retention["used_for_module3"]=retention.index.isin(CLASSES)
    retention.to_csv(REPORT/"selection_by_method.csv")
    split_counts=pd.crosstab(metadata.split,metadata.discoverymethod).reindex(index=["train","test"],columns=CLASSES)
    split_counts.to_csv(REPORT/"split_counts.csv")
    fig,ax=plt.subplots(figsize=(9,4.5))
    split_counts.plot.bar(stacked=True,color=COLORS,ax=ax,rot=0)
    ax.set(xlabel="Fixed split",ylabel="Number of planets",title="Discovery-method counts | disjoint name-proxy groups")
    ax.legend(title="Recorded class");save(fig,"split_counts")

    # Each measurement contributes one bin-token, hence each planet has six
    # non-negative integer events. Bins are learned inside each CV training fold.
    pipe=Pipeline([("bins",KBinsDiscretizer(encode="onehot-dense",strategy="quantile",subsample=None,quantile_method="linear")),
                   ("nb",MultinomialNB())])
    grid=GridSearchCV(pipe,{"bins__n_bins":[3,5,8],"nb__alpha":[.1,1.,10.]},
                      scoring="balanced_accuracy",cv=folds,n_jobs=1,refit=True,error_score="raise")
    grid.fit(x.iloc[train],y.iloc[train])
    best_nb=grid.best_estimator_
    pd.DataFrame(grid.cv_results_).to_csv(REPORT/"nb_cross_validation.csv",index=False)
    bins=best_nb.named_steps["bins"]
    names=bins.get_feature_names_out(FEATURES).tolist()
    for label,idx in [("train",train),("test",test)]:
        encoded=bins.transform(x.iloc[idx])
        assert (encoded>=0).all() and np.isin(encoded,[0,1]).all()
        assert np.all(encoded.sum(axis=1)==len(FEATURES))
        table=pd.DataFrame(encoded.astype(int),columns=names)
        table["discoverymethod"]=y.iloc[idx].to_numpy()
        table.to_csv(DATA/f"nb_{label}_counts.csv",index=False)
        table.head(12).to_csv(DATA/f"nb_{label}_sample.csv",index=False)
    encoded=pd.read_csv(DATA/"nb_train_counts.csv")
    preview(encoded[names[:6]+["discoverymethod"]],"nb_counts_preview","NB training counts | first six bin columns only; full numeric matrix linked")
    write_json({"requested_bins":int(grid.best_params_["bins__n_bins"]),
                "bin_edges":{f:edges.tolist() for f,edges in zip(FEATURES,bins.bin_edges_)},
                "actual_bin_counts":dict(zip(FEATURES,bins.n_bins_.astype(int).tolist())),
                "encoded_feature_names":names},"bin_edges.json")
    nb=best_nb.named_steps["nb"]
    write_json({"classes":nb.classes_.tolist(),"class_counts":nb.class_count_.tolist(),
                "class_log_prior":nb.class_log_prior_.tolist(),"alpha":float(nb.alpha),
                "feature_names":names,"feature_log_probabilities":nb.feature_log_prob_.tolist()},"nb_parameters.json")
    evidence=pd.DataFrame({"event":names,"log_P_RV_minus_log_P_Transit":nb.feature_log_prob_[0]-nb.feature_log_prob_[1]})
    evidence.to_csv(REPORT/"nb_event_evidence.csv",index=False)
    show=evidence.iloc[np.argsort(np.abs(evidence.iloc[:,1]))[-12:]].sort_values(evidence.columns[1])
    fig,ax=plt.subplots(figsize=(11,6))
    ax.barh(show.event,show.iloc[:,1],color=[COLORS[0] if v>0 else COLORS[1] for v in show.iloc[:,1]])
    ax.axvline(0,color="#555555",lw=1)
    ax.set(xlabel="Training log P(event | RV) - log P(event | Transit)",
           title="Twelve largest NB event contrasts | excludes class-prior contribution")
    save(fig,"nb_event_evidence")

    specs=[("tree_gini_depth2", "gini",2), ("tree_gini_depth3","gini",3), ("tree_entropy_depth4","entropy",4)]
    models={"multinomial_nb":best_nb}
    rows=[{"model":"multinomial_nb","cv_mean_balanced_accuracy":float(grid.best_score_),
           "cv_sd_balanced_accuracy":float(grid.cv_results_["std_test_score"][grid.best_index_])}]
    for name,criterion,depth in specs:
        tree=DecisionTreeClassifier(criterion=criterion,max_depth=depth,min_samples_leaf=20,random_state=SEED)
        cv=cross_val_score(tree,x.iloc[train],y.iloc[train],cv=folds,scoring="balanced_accuracy")
        tree.fit(x.iloc[train],y.iloc[train]);models[name]=tree
        rows.append({"model":name,"cv_mean_balanced_accuracy":float(cv.mean()),"cv_sd_balanced_accuracy":float(cv.std())})
        fig,ax=plt.subplots(figsize=({2:14,3:24,4:40}[depth],max(6,depth*3)))
        plot_tree(tree,feature_names=["Radius (Earth radii)","Mass (Earth masses)","Period (days)",
                                      "Star T (K)","Star M (solar masses)","Star R (solar radii)"],class_names=CLASSES,
                  filled=True,rounded=True,precision=3,fontsize=9,ax=ax)
        ax.set_title(f"{criterion.capitalize()} tree | max depth {depth}, minimum leaf 20 | training counts",fontsize=16)
        save(fig,name)
        (REPORT/f"{name}_rules.txt").write_text(export_text(tree,feature_names=FEATURES,decimals=5),encoding="utf-8")
        pd.DataFrame({"feature":FEATURES,"training_impurity_importance":tree.feature_importances_}).to_csv(REPORT/f"{name}_importance.csv",index=False)
    threshold=float(models["tree_gini_depth2"].tree_.threshold[0])
    left=x.iloc[train].pl_orbper<=threshold
    node_counts={"parent":[int((y.iloc[train]==c).sum()) for c in CLASSES],
                 "left":[int((y.iloc[train][left]==c).sum()) for c in CLASSES],
                 "right":[int((y.iloc[train][~left]==c).sum()) for c in CLASSES]}
    entropy=lambda counts:float(-sum(p*np.log2(p) for p in np.asarray(counts)/sum(counts) if p>0))
    gini=lambda counts:float(1-sum((np.asarray(counts)/sum(counts))**2))
    worked={"feature":"pl_orbper","threshold_days":threshold,"class_order":CLASSES,
            "counts":node_counts,"entropy":{n:entropy(c) for n,c in node_counts.items()},
            "gini":{n:gini(c) for n,c in node_counts.items()}}
    for key,func in [("information_gain_bits",entropy),("gini_decrease",gini)]:
        worked[key]=func(node_counts["parent"])-sum(sum(node_counts[c])/len(train)*func(node_counts[c]) for c in ["left","right"])
    write_json(worked,"worked_split.json")
    comparison=pd.DataFrame(rows)
    # Selection is sealed here, BEFORE any test labels are scored.
    selected_tree=comparison[comparison.model.str.startswith("tree_")].sort_values(
        ["cv_mean_balanced_accuracy","model"],ascending=[False,True]).iloc[0].model
    models["majority_baseline"]=DummyClassifier(strategy="most_frequent").fit(x.iloc[train],y.iloc[train])
    output={}
    predictions=metadata.iloc[test].reset_index(drop=True).copy()
    for name,model in models.items():
        output[name]={"train":metrics(model,x.iloc[train],y.iloc[train]),"test":metrics(model,x.iloc[test],y.iloc[test])}
        predictions[name]=model.predict(x.iloc[test])
    for row in rows:
        name=row["model"]
        row.update({"train_accuracy":output[name]["train"]["accuracy"],"test_accuracy":output[name]["test"]["accuracy"],
                    "train_balanced_accuracy":output[name]["train"]["balanced_accuracy"],
                    "test_balanced_accuracy":output[name]["test"]["balanced_accuracy"]})
    comparison=pd.DataFrame(rows);comparison.to_csv(REPORT/"model_comparison.csv",index=False)
    predictions.to_csv(REPORT/"test_predictions.csv",index=False)
    for name in ["multinomial_nb",selected_tree]:
        output[name]["conditional_group_bootstrap"]=bootstrap_interval(y.iloc[test].to_numpy(),predictions[name].to_numpy(),groups.iloc[test].to_numpy())
    # A prespecified sensitivity check removes possibly derived mass; parameters
    # stay fixed. This does not select the primary model or change its feature set.
    sensitivity={}
    for name in ["multinomial_nb",selected_tree]:
        model=clone(models[name]).fit(x.iloc[train].drop(columns="pl_bmasse"),y.iloc[train])
        sensitivity[name]=metrics(model,x.iloc[test].drop(columns="pl_bmasse"),y.iloc[test])
    fig,ax=plt.subplots(figsize=(7,5.5));confusion_plot(ax,output["multinomial_nb"]["test"]["confusion_matrix"],"Multinomial NB | untouched holdout");save(fig,"nb_confusion")
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for ax,(name,criterion,depth) in zip(axes,specs):
        confusion_plot(ax,output[name]["test"]["confusion_matrix"],f"{criterion.capitalize()}, depth {depth}")
    fig.suptitle("Three prespecified trees | same holdout; selection uses training CV only",y=1.04)
    fig.tight_layout();save(fig,"tree_confusions")
    fig,ax=plt.subplots(figsize=(11,5))
    ids=np.arange(len(comparison))
    ax.bar(ids-.2,comparison.cv_mean_balanced_accuracy,.4,yerr=comparison.cv_sd_balanced_accuracy,label="Training CV mean +/- fold SD",color=COLORS[1])
    ax.bar(ids+.2,comparison.test_balanced_accuracy,.4,label="Final holdout",color=COLORS[0])
    ax.axhline(.5,color="#777777",ls="--",label="Single-class balanced accuracy")
    ax.set(xticks=ids,xticklabels=["Multinomial NB","Gini depth 2","Gini depth 3","Entropy depth 4"],
           ylabel="Balanced accuracy (mean class recall)",ylim=(0,1.06),title="Model comparison | no test-based tuning")
    ax.legend(loc="lower right");save(fig,"model_comparison")
    profile=frame.assign(split=metadata.split).groupby(["split","discoverymethod"])[FEATURES].median()
    profile.to_csv(REPORT/"physical_medians.csv")
    prep={"source":"data/processed/exoplanets_clean.csv","source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
          "source_rows":len(raw),"target_method_rows":int(is_target.sum()),"eligible_rows":len(frame),
          "excluded_target_rows":int((is_target&~valid).sum()),"other_methods_excluded":int((~is_target).sum()),
          "features":FEATURES,"units":dict(zip(FEATURES,UNITS)),"target":"discoverymethod", "classes":CLASSES,
          "train_n":len(train),"test_n":len(test),"train_groups":int(groups.iloc[train].nunique()),"test_groups":int(groups.iloc[test].nunique()),
          "row_overlap":0,"planet_name_overlap":0,"name_proxy_group_overlap":0,
          "group_proxy":"pl_name with final space + lowercase b-z removed; not verified hostname",
          "unchanged_name_groups":int((frame.host_proxy==frame.pl_name).sum()),
          "outer_split":"first of five stratified group folds; seed 42","inner_cv":"five stratified group folds within train; seed 43",
          "preprocessing":"positive complete cases; no imputation, outlier trimming or global scaling; quantile bins fitted inside NB training folds",
          "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__}
    write_json(prep,"preparation.json")
    result={"selected_nb_params":{k:float(v) if isinstance(v,float) else int(v) for k,v in grid.best_params_.items()},
            "selected_tree":selected_tree,"selection_metric":"mean training-CV balanced accuracy; test never used for selection",
            "models":output,"mass_omission_sensitivity":sensitivity}
    write_json(result,"results.json")
    print(json.dumps({"preparation":prep,"comparison":rows,"selected_tree":selected_tree,
                      "nb_holdout":output['multinomial_nb']['test'],"tree_holdout":output[selected_tree]['test'],
                      "mass_omission":sensitivity},indent=2))


if __name__=="__main__":
    with threadpool_limits(limits=1):run()
