import csv, itertools, hashlib
from pathlib import Path

out_path = Path("synthetic_large_pipeline_all_combinations_1based.csv")

# ---------- strategy spaces ----------
sampling = ['full','random','snapshot','stratified']
invalid_value = ['none','sentinel','regex','both']
missing_value = ['drop','mean','median','most_frequent','knn']
normalization = ['none','ss','rs','ma','mm']
distribution_shape = ['none','log1p','sqrt','boxcox','yeojohnson']
floating_point = ['none','snap','round','both']
multicollinearity = ['none','drop_high_vif']
poly_pca = ['none','pca','sparsepca','minibatchsparsepca','kernelpca']
stopword = ['none','sw']
whitespace = ['none','wc']
punctuation = ['none','sc']
lowercase = ['none','lc']
model = ['lr']
outlier = ['none','if','iqr','lof']

def enc(lst): return {v:i+1 for i,v in enumerate(lst)}

E = {
 'sampling':enc(sampling),'invalid_value':enc(invalid_value),
 'missing_value':enc(missing_value),'normalization':enc(normalization),
 'distribution_shape':enc(distribution_shape),'floating_point':enc(floating_point),
 'multicollinearity':enc(multicollinearity),'poly_pca':enc(poly_pca),
 'stopword':enc(stopword),'whitespace':enc(whitespace),
 'punctuation':enc(punctuation),'lowercase':enc(lowercase),
 'model':enc(model),'outlier':enc(outlier)
}

pipeline_cols = [
 "sampling","invalid_value","missing_value","normalization",
 "distribution_shape","floating_point","multicollinearity","poly_pca",
 "stopword","whitespace","punctuation","lowercase","model"
]

profile_cols = [
 "outlier_bef_normalization_strat","outlier_bef_outlier_strat",
 "diff_sensitive_attr","ratio_sensitive_attr","cov","class_imbalance_ratio",
 "corr_Age","ot_Age","corr_Workclass","ot_Workclass","corr_fnlwgt","ot_fnlwgt",
 "corr_Education","ot_Education","corr_Education_Num","ot_Education_Num",
 "corr_Martial_Status","ot_Martial_Status","corr_Occupation","ot_Occupation",
 "corr_Relationship","ot_Relationship","corr_Race","ot_Race","corr_Sex","ot_Sex",
 "corr_Capital_Gain","ot_Capital_Gain","corr_Capital_Loss","ot_Capital_Loss",
 "corr_Hours_per_week","ot_Hours_per_week","corr_Country","ot_Country",
 "utility_sp"
]

def U01(*x):
    h = hashlib.blake2b("|".join(map(str,x)).encode(),digest_size=8).digest()
    return (int.from_bytes(h,"big")%10_000_000)/10_000_000

def U11(*x): return 2*U01(*x)-1

with out_path.open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(pipeline_cols+profile_cols)

    for combo in itertools.product(
        sampling,invalid_value,missing_value,normalization,
        distribution_shape,floating_point,multicollinearity,poly_pca,
        stopword,whitespace,punctuation,lowercase,model):

        row=[E[c][v] for c,v in zip(pipeline_cols,combo)]

        ob1 = E['outlier'][outlier[int(U01(*combo,'o1')*4)%4]]
        ob2 = E['outlier'][outlier[int(U01(*combo,'o2')*4)%4]]

        profiles=[
          ob1,ob2,
          round(U11(*combo,'d'),6),
          round(U01(*combo,'r'),6),
          round(0.05+0.95*U01(*combo,'c'),6),
          round(0.1+9.9*U01(*combo,'imb'),6)
        ]

        for k in ["Age","Workclass","fnlwgt","Education","Education_Num",
                  "Martial_Status","Occupation","Relationship","Race","Sex",
                  "Capital_Gain","Capital_Loss","Hours_per_week","Country"]:
            profiles += [round(U11(*combo,k),6), round(U01(*combo,'ot'+k),6)]

        utility = round(0.05 + 0.20 * U01(*combo, 'u'), 6)
        w.writerow(row+profiles+[utility])
