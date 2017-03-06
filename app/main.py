from os.path import dirname, join

import numpy as np
import pandas as pd
import pymongo

from bokeh.plotting import figure
from bokeh import palettes
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, Div
from bokeh.models.widgets import Slider, Select, TextInput
from bokeh.io import curdoc

client = pymongo.MongoClient()
db = client.fantasy_football
player_history = db.player_data.find({},{"history":1})
player_dfs = {}
for i, player in enumerate(player_history):
    df = pd.DataFrame(player["history"])
    df.index = df["round"]
    player_dfs[i] = df

player_details = pd.DataFrame(db.player_details.find_one({},{"elements":1})["elements"])
player_details.index = player_details.id

axis_variables = [
    'yellow_cards',
    'penalties_saved',
    'dreamteam_count',
    'transfers_in',
    'chance_of_playing_this_round',
    'saves',
    'creativity',
    'value_form',
    'squad_number',
    'event_points',
    'own_goals',
    'now_cost',
    'ep_next',
    'clean_sheets',
    'form',
    'transfers_in_event',
    'transfers_out',
    'selected_by_percent',
    'element_type',
    'team',
    'in_dreamteam',
    'assists',
    'points_per_game',
    'cost_change_event',
    'red_cards',
    'ea_index',
    'ep_this',
    'goals_conceded',
    'cost_change_start_fall',
    'chance_of_playing_next_round',
    'goals_scored',
    'minutes',
    'value_season',
    'bonus',
    'total_points',
    'threat',
    'bps',
    'transfers_out_event',
    'cost_change_start',
    'influence',
    'penalties_missed',
    'cost_change_event_fall'
]

axis_names = {
    name: name.capitalize().replace("_", " ").replace("event", "gameweek") for name in axis_variables
}
axis_names["team"] = "Team ID"
axis_names["now_cost"] = "Price (times 10)"
axis_names["element_type"] = "Position ID"

desc = Div(text=open(join(dirname(__file__), "description.html")).read(), width=800)

# Create Input controls
minutes = Slider(title="Minimum number of minutes played", value=900, start=0, end=3420, step=30)
#start_week = Slider(title="Gameweek start", start=1, end=38, value=1, step=1)
#end_week = Slider(title="Gameweek end", start=1, end=38, value=1, step=1)
#club = Select(title="Club", value="All",
#              options=open(join(dirname(__file__), 'clubs.txt')).read().split())
#position = Select(title="Position", value="All",
#                  options=open(join(dirname(__file__), 'positions.txt')).read().split())
x_axis = Select(title="X Axis", options=sorted(axis_names.keys()), value="total_points")
y_axis = Select(title="Y Axis", options=sorted(axis_names.keys()), value="now_cost")

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[]))

hover = HoverTool(tooltips=[
    ("Name", "@name"),
    ("X-Value", "@x"),
    ("Y-Value", "@y")
])

p = figure(plot_height=600, plot_width=700, title="", toolbar_location=None, tools=[hover])
p.circle(x="x", y="y", source=source, size=7)#, color="color", line_color=None, fill_alpha="alpha")


def update():
    df = player_details[
        (player_details["minutes"] > minutes.value)
    ]
    # if club != "All":
    #     df = player_details[
    #         (player_details["club"] > club)
    #     ]

    #colours = palettes.Spectral5
    #c = "#31AADE"
    #if color.value != 'None':
    #    groups = pd.qcut(df[color.value].values, len(colours))
    #    c = [colours[xx] for xx in groups.codes]
    # colour = df[z_axis.value]
    
    p.xaxis.axis_label = axis_names[x_axis.value]
    p.yaxis.axis_label = axis_names[y_axis.value]
    p.title.text = "{} players selected".format(df.shape[0])
    source.data = dict(
        x=df[x_axis.value],
        y=df[y_axis.value],
        # color=colour,
        name=df["first_name"] + " " + df["second_name"],
    )

controls = [minutes, x_axis, y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

sizing_mode = 'fixed'

inputs = widgetbox(*controls, sizing_mode=sizing_mode)
l = layout([
    [desc],
    [inputs, p],
], sizing_mode=sizing_mode)

update()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "Players"
