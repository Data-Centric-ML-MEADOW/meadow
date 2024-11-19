import pandas as pd
import plotly.graph_objects as go

# Create the DataFrame
data = {
    "Model": [
        "ResNet 152", "ResNet 50", "ResNet 34",
        "Snapshot Ensemble", 
        "SE 50, 3 experts", "SE 50, 3 experts", "SE 34, 5 experts", "SE 34, 3 experts",
        "Mixture of Experts (MoE)", 
        "MoE 50, 3 experts", "MoE 34, 5 experts", "MoE 34, 3 experts",
        "Vision Transformer (ViT)", 
        "ViT 16 No Freeze", "ViT 16 Freeze"
    ],
    "Test Accuracy": [
        0.721133, 0.738193, 0.682059, None,
        0.677526, 0.677315, 0.642495, 0.640672,
        None,
        0.681311, 0.647613, 0.678951,
        None,
        0.561871, 0.448202
    ],
    "Test Recall": [
        0.272167, 0.260434, 0.223976, None,
        0.255944, 0.262979, 0.227946, 0.226629,
        None,
        0.266920, 0.222860, 0.248945,
        None,
        0.196435, 0.098078
    ],
    "Test F1": [
        0.275299, 0.266368, 0.233204, None,
        0.251981, 0.257199, 0.223529, 0.221488,
        None,
        0.260069, 0.218446, 0.245326,
        None,
        0.169430, 0.088369
    ]
}

df = pd.DataFrame(data)

# Define colors
header_color = 'lightblue'
subheader_color = 'lightgrey'
cell_color = 'white'
nan_color = 'white'

# Prepare the table for Plotly
fig = go.Figure(data=[go.Table(
    header=dict(
        values=["<b>Model</b>", "<b>Test Accuracy</b>", "<b>Test Recall</b>", "<b>Test F1</b>"],
        fill_color=header_color,
        align='center',
        font=dict(size=12, color='black')
    ),
    cells=dict(
        values=[
            df["Model"], 
            df["Test Accuracy"].fillna(''),  # Replace NaN with an empty string
            df["Test Recall"].fillna(''),
            df["Test F1"].fillna('')
        ],
        fill_color=[
            [subheader_color if "Snapshot" in model or "Mixture" in model or "Vision" in model else cell_color for model in df["Model"]],
            cell_color, cell_color, cell_color
        ],
        align='center',
        font=dict(size=11, color='black')
    )
)])

# Show table
fig.update_layout(
    title=dict(
        text="Experimental Results Table",
        x=0.5,
        font=dict(size=16)
    ),
    margin=dict(l=10, r=10, t=30, b=10)
)

# Export table as an image or HTML
fig.write_image("results_table_colored.png")
fig.write_html("results_table_colored.html")
fig.show()
