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
# TODO only grab latest data. currently duplicating.
player_history = db.player_data.find({},{"history":1})
player_dfs = {}
for i, player in enumerate(player_history):
    if i < 649:
        continue
    df = pd.DataFrame(player["history"])
    df.index = df["round"]
    player_dfs[i] = df

player_panel = pd.Panel(player_dfs)

player_details = pd.DataFrame(db.player_details.find_one({},{"elements":1})["elements"])
player_details.index = player_details.id

axis_variables = ['assists', 'attempted_passes', 'big_chances_created',
 'big_chances_missed', 'bonus', 'bps', 'clean_sheets',
 'clearances_blocks_interceptions', 'completed_passes', 'creativity',
 'dribbles', 'ea_index', 'element', 'errors_leading_to_goal',
 'errors_leading_to_goal_attempt', 'fixture', 'fouls', 'goals_conceded',
 'goals_scored', 'ict_index', 'id', 'influence', 'key_passes',
 'kickoff_time', 'kickoff_time_formatted', 'loaned_in', 'loaned_out',
 'minutes', 'offside', 'open_play_crosses', 'opponent_team', 'own_goals',
 'penalties_conceded', 'penalties_missed', 'penalties_saved',
 'recoveries', 'red_cards', 'round', 'saves', 'selected', 'tackled',
 'tackles', 'target_missed', 'team_a_score', 'team_h_score', 'threat',
 'total_points', 'transfers_balance', 'transfers_in', 'transfers_out',
 'value', 'was_home', 'winning_goals', 'yellow_cards']

axis_names = {
    name: name.capitalize().replace("_", " ").replace("event", "gameweek") for name in axis_variables
}

desc = Div(text=open(join(dirname(__file__), "description.html")).read(), width=800)

# Create Input controls
#minutes = Slider(title="Minimum number of minutes played", value=900, start=0, end=3420, step=30)
start = Slider(title="Gameweek start", start=1, end=38, value=1, step=1)
end = Slider(title="Gameweek end", start=1, end=38, value=38, step=1)
#club = Select(title="Club", value="All",
#              options=open(join(dirname(__file__), 'clubs.txt')).read().split())
#position = Select(title="Position", value="All",
#                  options=open(join(dirname(__file__), 'positions.txt')).read().split())
x_axis = Select(title="X Axis", options=sorted(axis_names.keys()), value="goals_scored")
y_axis = Select(title="Y Axis", options=sorted(axis_names.keys()), value="assists")

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
    df = player_panel.loc[:, start.value:end.value, :]

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
        x=df.loc[:, :, x_axis.value].sum(),
        y=df.loc[:, :, y_axis.value].sum(),
        # color=colour,
        name=player_details.loc[df.loc[:,:,"element"].max(), "first_name"] + " " +\
             player_details.loc[df.loc[:,:,"element"].max(), "second_name"],
    )

controls = [start, end, x_axis, y_axis]
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
