import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import gradio as gr

np.random.seed(42)

cell_lines = [f"CellLine_{i}" for i in range(100)]
genes = [f"Gene_{j}" for j in range(500)]
drugs = ["DrugA", "DrugB", "DrugC"]
selected_drug = "DrugA"

expr_df = pd.DataFrame(
    np.random.rand(100, 500) * 10,
    index=cell_lines,
    columns=genes
)

rows = []
for drug in drugs:
    for cl in cell_lines:
        gene_mean = np.mean(expr_df.loc[cl])
        ic50 = np.random.normal(5, 1.2) - 0.01 * gene_mean
        rows.append([cl, drug, ic50])

drug_df = pd.DataFrame(rows, columns=["CELL_LINE", "DRUG", "IC50"])

sub = drug_df[drug_df["DRUG"] == selected_drug]
data = expr_df.join(sub.set_index("CELL_LINE")["IC50"], how="inner")

ic50_median = data["IC50"].median()
data["IC50_status"] = np.where(
    data["IC50"] >= ic50_median,
    f"{selected_drug}_Direncli",
    f"{selected_drug}_Hassas"
)

binary_df = pd.DataFrame(index=data.index)
for gene in genes:
    gene_median = data[gene].median()
    binary_df[f"{gene}_Yuksek"] = data[gene] >= gene_median

status_dummies = pd.get_dummies(data["IC50_status"], prefix="", prefix_sep="").astype(bool)
final_transaction_df = binary_df.join(status_dummies)

def run_apriori_analysis(min_support_input, min_confidence_input):
    if min_support_input < 0.05:
        return ("Hata: Destek eþiði çok düþük.", None)
    try:
        frequent_itemsets = apriori(
            final_transaction_df,
            min_support=min_support_input,
            use_colnames=True
        )

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence_input
        )

        drug_status_items = [f"{selected_drug}_Direncli", f"{selected_drug}_Hassas"]
        rules_filtered = rules[
            rules["consequents"].apply(lambda x: any(item in x for item in drug_status_items))
        ].sort_values(by="confidence", ascending=False)

        if rules_filtered.empty:
            return (f"Kural bulunamadý.", None)

        markdown_result = (
            f"## Apriori Sonuçlarý ({selected_drug})\n"
            f"Destek: {min_support_input:.2f}, Güven: {min_confidence_input:.2f}\n\n"
            f"Toplam kural: **{len(rules_filtered)}**\n"
        )

        result_rows = []
        for _, row in rules_filtered.head(10).iterrows():
            antecedents_str = " ^ ".join([str(x) for x in row["antecedents"]])
            consequents_str = " -> ".join([str(x) for x in row["consequents"]])
            result_rows.append({
                "Kural": f"{antecedents_str} => {consequents_str}",
                "Destek": f"{row['support']:.3f}",
                "Güven": f"{row['confidence']:.3f}"
            })

        results_df = pd.DataFrame(result_rows)
        return markdown_result, results_df.to_html(index=False)

    except Exception as e:
        return (f"Hata: {e}", None)

min_support_slider = gr.Slider(0.05, 0.50, step=0.01, value=0.25, label="Minimum Destek")
min_confidence_slider = gr.Slider(0.50, 1.00, step=0.01, value=0.60, label="Minimum Güven")

output_text = gr.Markdown()
output_html = gr.HTML()

iface = gr.Interface(
    fn=run_apriori_analysis,
    inputs=[min_support_slider, min_confidence_slider],
    outputs=[output_text, output_html],
    title="Farmakogenomik Apriori Analizi",
    description="Ýlaç hassasiyeti ve direnç iliþkileri için birliktelik kurallarý keþfi."
)

if __name__ == "__main__":
    iface.launch()
