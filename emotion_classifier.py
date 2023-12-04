import numpy as np
import pandas as pd
from transformers import AutoTokenizer, RobertaForSequenceClassification
import torch
import plotly.express as px

from preproc import preproc_class
from dist_list import emotion_dist

model = RobertaForSequenceClassification.from_pretrained("model/", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("model/", local_files_only=True)

def find_key(input_dict, value):
    for key, val in input_dict.items():
        if val == value:
            return key
    return "None"

def perform_emotion_classification(input_text):
    input_ids = torch.tensor([
        tokenizer.encode(
            preproc_class(lst=[input_text]).preprocessing([1, 2, 3, 4, 5, 6])[0]
        )
    ])
    with torch.no_grad():
        out = model(input_ids)
        result = out.logits.softmax(dim=-1).tolist()

        predicted_emotion = find_key(emotion_dist, np.argmax(result[0]))
        confidence_score = f"{round(max(result[0]) * 100, 2)}%"

    emotions = ["Enjoyment", "Fear", "Anger", "Sadness", "Disgust", "Surprise", "Other"]
    percentages = [round(score * 100, 2) for score in result[0]]

    plot_data = {'Emotion': emotions, 'Percentage': percentages}
    df = pd.DataFrame(plot_data)

    color_map = {
            "Enjoyment": "#FF1C99", "Fear": "#4D08A1", "Anger": "#E74C3C",
            "Sadness": "#72809D", "Disgust": "#58D68D", "Surprise": "#F4D03F", "Other": "#BDC3C7"
        }
    fig = px.bar(df, x='Emotion', y='Percentage', text='Percentage',
                 labels={'Emotion': 'Emotion', 'Percentage': 'Percentage'},
                 height=400, width=670, color='Emotion', color_discrete_map=color_map)
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

    # Adjust plot styling to match the website's background color and font
    fig.update_layout(
        plot_bgcolor='#161B22',        # Plot background color
        paper_bgcolor='#161B22',       # Paper (canvas) background color
        font_color='#C9D1D9',          # Font color
        xaxis=dict(linecolor='#C9D1D9', showgrid=False),   # X-axis styling
        yaxis=dict(linecolor='#C9D1D9', showgrid=False),   # Y-axis styling
        font_family='Inter',           # Use the 'Inter' font
        legend=dict(bgcolor='#161B22', bordercolor='rgba(0,0,0,0)', font=dict(color='#C9D1D9')) # Legend styling
    )

    # Remove hover label boxes and adjust hover font
    fig.update_traces(
        hoverinfo='x+y', hoverlabel=dict(bgcolor='#161B22', bordercolor='#161B22', font=dict(family='Inter', color='#C9D1D9'))
    )

    return predicted_emotion, confidence_score, fig
