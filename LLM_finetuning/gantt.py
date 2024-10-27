import plotly.figure_factory as ff
import plotly.graph_objects as go

# Define the data
df = [dict(Task="Job A  ", Start='1', Finish='2', Resource='GOLD'),
      dict(Task="Job B  ", Start='3', Finish='4', Resource='SILVER'),
      dict(Task="Job C  ", Start='2', Finish='5', Resource='BRONZE')]

# Define colors for the resources
colors = dict(GOLD='rgb(255, 140, 0)', SILVER='rgb(192, 192, 192)', BRONZE='rgb(205, 127, 50)')

# Create the Gantt chart
fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True)

# Customize the x-axis to display abstract labels
fig.update_xaxes(
    tickvals=[1, 2, 3, 4, 5],  # Define the ticks
    ticktext=["t1", "t2", "t3", "t4", "t5"],  # Define custom tick labels
)

# Add padding between the task names and plot
fig.update_yaxes(title_standoff=50)  # Increase standoff for padding


# Remove unnecessary interactive selector (optional: this depends on usage context and visualization setup)
fig.update_layout(
    xaxis={'rangeselector':{'visible':False}},
    autosize=False,
    width=1000,
    height=500,
    margin=go.layout.Margin(
        l=50,
        r=5,
        b=5,
        t=30,
        pad=1
    ),
    # paper_bgcolor="LightSteelBlue",
)

# fig['layout']['yaxis']['autorange'] = True
fig.update_yaxes(
    title_text=" ",   # Add a blank title to increase space
    automargin=True   # Enable automargin to allow for extra spacing
)

# Save the figure as PDF
fig.write_image("gantt.png")
# fig.show()
