#!/usr/bin/env python
# coding: utf-8

# In[22]:


import json
import pandas as pd
import requests
from scipy import stats
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import plotly.express as px
from numpy import nan
import geopandas
import matplotlib.pyplot as plt
import folium
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import streamlit as st
import plotly.figure_factory as ff


# In[2]:


key = '91b8563b-459b-4816-9417-2e3860f1f3e4'
url = ' https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=11063&compact=true&verbose=false'
    
json_data = requests.get(url, params={'key': key}).json()
laadpaallocaties = pd.DataFrame.from_dict(json_data)


laadpaaldata = pd.read_csv('laadpaaldata.csv')


response = requests.get("https://opendata.rdw.nl/resource/m9d7-ebf2.json")
voertuigen_data = pd.DataFrame.from_dict(response.json())


# In[3]:


laadpaaldata['Started'] = pd.to_datetime(laadpaaldata['Started'], errors='coerce')
laadpaaldata['Ended'] = pd.to_datetime(laadpaaldata['Ended'], errors='coerce')



#missing values droppen

laadpaaldata_dt = laadpaaldata.dropna()



#alles waar het verschil tussen 'Ended' en 'Started' droppen

laadpaaldata_dt_time = laadpaaldata_dt[laadpaaldata_dt['Ended'] - laadpaaldata_dt['Started'] >= '0 days 00:00:00']



#alles waar 'ConnectedTime' en 'ChargeTime' minder is dan 0 droppen
laadpaaldata_dt_1 = laadpaaldata_dt_time[laadpaaldata_dt_time['ConnectedTime'] >= 0]
laadpaaldata_dt_2 = laadpaaldata_dt_time[laadpaaldata_dt_time['ChargeTime'] >= 0]



#uitschieters droppen van 'ConnectedTime' en 'ChargeTime'

laadpaaldata_dt_3 = laadpaaldata_dt_2[(np.abs(stats.zscore(laadpaaldata_dt_2['ConnectedTime'])) < 3)]
laadpaaldata_dt_4 = laadpaaldata_dt_3[(np.abs(stats.zscore(laadpaaldata_dt_3['ChargeTime'])) < 3)]


# In[4]:


laadpaallocaties.drop(['GeneralComments','OperatorsReference','MetadataValues','DateLastConfirmed'], axis = 1)


# In[5]:


voertuigen_data.drop(['aanhangwagen_autonoom_geremd','aanhangwagen_middenas_geremd','maximale_constructiesnelheid_brom_snorfiets','vermogen_brom_snorfiets','europese_voertuigcategorie_toevoeging','europese_uitvoeringcategorie_toevoeging','vervaldatum_tachograaf','type_gasinstallatie','oplegger_geremd'], axis =1)


# In[6]:


variables_of_intrest = ['AddressLine1', 'Town', 'StateOrProvince', 'Postcode', 'Latitude', 'Longitude']


def get_geo_data(dict_, variables_of_intrest):
    values = []
    for variable in variables_of_intrest:
        if variable in dict_.keys():
            if dict_[variable] == "":
                values.append(np.NaN)
            else:
                values.append(dict_[variable])
        else:
            values.append(np.NaN)
    return values

geo_data = []
for index, row in laadpaallocaties.iterrows():
    values = get_geo_data(row['AddressInfo'], variables_of_intrest=variables_of_intrest)
    geo_data.append(values)
    
geo_df = pd.DataFrame(geo_data, columns = variables_of_intrest)
geo_df = geo_df.dropna()
geo_df


# In[7]:


def color_producer(type):
    if type == 'North Brabant':
        return 'goldenrod'
    elif type == 'Samenwerkingsverband Regio Eindhoven':
        return 'goldenrod'
    elif type == 'Noord-Brabant':
        return 'goldenrod'
    elif type == 'Nordbraban':
        return 'goldenrod'
    elif type == 'Noord Brabant ':
        return 'goldenrod'
    elif type == 'South Holland':
        return 'Orange'
    elif type == 'Zuid-Holland':
        return 'Orange'
    elif type == 'Zuid Holland':
        return 'Orange'
    elif type == 'ZH':
        return 'Orange'
    elif type == 'North Holland':
        return 'Yellow'
    elif type == 'Stadsregio Amsterdam':
        return 'Yellow'
    elif type == 'Noord-Holland':
        return 'Yellow'
    elif type == 'Nordholland':
        return 'Yellow'
    elif type == 'Noord Holand':
        return 'Yellow'
    elif type == 'Noord Holland':
        return 'yellow'
    elif type == 'Noord-Hooland':
        return 'yellow'
    elif type == 'Zeeland':
        return 'aqua'
    elif type == 'Seeland':
        return 'aqua'
    elif type == 'Utrecht':
        return 'Navy'
    elif type == 'UT':
        return 'navy'
    elif type == 'UTRECHT':
        return 'navy'
    elif type == 'Limburg':
        return 'red' 


# In[8]:


gdf = geopandas.GeoDataFrame(
    geo_df, geometry=geopandas.points_from_xy(geo_df.Longitude, geo_df.Latitude))


# In[9]:


key = '91b8563b-459b-4816-9417-2e3860f1f3e4'
url = ' https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=11063&compact=true&verbose=false'
    
json_data = requests.get(url, params={'key': key}).json()
laadpaallocaties = pd.DataFrame.from_dict(json_data)


laadpaaldata = pd.read_csv('laadpaaldata.csv')


response = requests.get("https://opendata.rdw.nl/resource/m9d7-ebf2.json")
voertuigen_data = pd.DataFrame.from_dict(response.json())


# In[10]:


#st.text("Op deze kaart worden de laadpalen weergegeven uit de dataframe. De laadpalen zijn groepeerd per provincie,
#dankzij de kleuren legenda kan je in één oogopslag zien welke kleur bij welke provincie hoort. Hier is terug te zien dat de meeste laadpalen zich bevinden in de randstad. Dit is ook niet zo gek want het voornaamste gedeelte van de bevolking woont in de randstad.")

m = folium.Map(location=[52.0907374,5.1214209], zoom_start=7.5)

for Town in gdf.iterrows():
    row_values = Town[1]
    location = [row_values['Latitude'], row_values['Longitude']]
    marker = folium.Circle(location=location, popup=row_values['AddressLine1'], color=color_producer(row_values['StateOrProvince']),
    fill_color=color_producer(row_values['StateOrProvince'])
    )
    marker.add_to(m)


# In[11]:


#  ik heb een functie gevonden op het internet voor het toevoegen van een categorische legenda:
# (bron: https://stackoverflow.com/questions/65042654/how-to-add-categorical-legend-to-python-folium-map)

def add_categorical_legend(folium_map, title, colors, labels):
    if len(colors) != len(labels):
        raise ValueError("colors and labels must have the same length.")

    color_by_label = dict(zip(labels, colors))
    
    legend_categories = ""     
    for label, color in color_by_label.items():
        legend_categories += f"<li><span style='background:{color}'></span>{label}</li>"
        
    legend_html = f"""
    <div id='maplegend' class='maplegend'>
      <div class='legend-title'>{title}</div>
      <div class='legend-scale'>
        <ul class='legend-labels'>
        {legend_categories}
        </ul>
      </div>
    </div>
    """
    script = f"""
        <script type="text/javascript">
        var oneTimeExecution = (function() {{
                    var executed = false;
                    return function() {{
                        if (!executed) {{
                             var checkExist = setInterval(function() {{
                                       if ((document.getElementsByClassName('leaflet-top leaflet-right').length) || (!executed)) {{
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.display = "flex"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.flexDirection = "column"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].innerHTML += `{legend_html}`;
                                          clearInterval(checkExist);
                                          executed = true;
                                       }}
                                    }}, 100);
                        }}
                    }};
                }})();
        oneTimeExecution()
        </script>
      """
   

    css = """

    <style type='text/css'>
      .maplegend {
        z-index:9999;
        float:right;
        background-color: rgba(255, 255, 255, 1);
        border-radius: 5px;
        border: 2px solid #bbb;
        padding: 10px;
        font-size:12px;
        positon: relative;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 0px solid #ccc;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    """

    folium_map.get_root().header.add_child(folium.Element(script + css))

    return folium_map


# In[15]:


st.text("Op deze kaart worden de laadpalen weergegeven uit de dataframe. De laadpalen zijn groepeerd per provincie, dankzij de kleuren legenda kan je in één oogopslag zien welke kleur bij welke provincie hoort. Hier is terug te zien dat de meeste laadpalen zich bevinden in de randstad. Dit is ook niet zo gek want het voornaamste gedeelte van de bevolking woont hier.")

m = add_categorical_legend(m, 'StateOrProvince',
colors = ['goldenrod', 'Orange', 'yellow', 'aqua', 'navy', 'red'],
labels = ['Noord-Brabant', 'Zuid-Holland', 'Noord-Holland', 'Zeeland', 'Utrecht', 'Limburg'])

folium_static(m)


# In[23]:


#gemiddelde en mediaan berekenen
contimemean = laadpaaldata_dt_4['ConnectedTime'].mean()
chatimemean = laadpaaldata_dt_4['ChargeTime'].mean()

contimemedian = laadpaaldata_dt_4['ConnectedTime'].median()
chatimemedian = laadpaaldata_dt_4['ChargeTime'].median()

#displot creëren
fig = ff.create_distplot([laadpaaldata_dt_4['ConnectedTime'], laadpaaldata_dt_4['ChargeTime']], 
                         group_labels=['Tijd aan de lader', 'Tijd om op te laden'], show_rug=False, curve_type='normal')

#verticale lijnen van gemiddelde en mediaan toevoegen
fig.add_shape(type='line', x0=contimemean, y0=0, x1=contimemean, y1=1, line=dict(color='Blue',), xref='x', yref='paper', 
              name='Gemiddelde Connected time')
fig.add_shape(type='line', x0=chatimemean, y0=0, x1=chatimemean, y1=1, line=dict(color='Red',), xref='x', yref='paper',
             name='Gemiddelde Charge time')
fig.add_shape(type='line', x0=contimemedian, y0=0, x1=contimemedian, y1=1, line=dict(color='Blue',), xref='x', yref='paper',
             name='Mediaan Connected time')
fig.add_shape(type='line', x0=chatimemedian, y0=0, x1=chatimemedian, y1=1, line=dict(color='Red',), xref='x', yref='paper',
             name='Mediaan Charge time')

#annotations bij de lijnen toevoegen
fig.add_annotation(x=contimemean, y=0.8, yref='paper',
            text="Gemiddelde tijd aan de lader",
            showarrow=True, ax=120)
fig.add_annotation(x=chatimemean, y=0.9, yref='paper',
            text="Gemiddelde tijd om op te laden",
            showarrow=True,ax=150, ay=-60)
fig.add_annotation(x=contimemedian, y=0.6, yref='paper',
            text="Mediaan tijd aan de lader",
            showarrow=True, ax=120)
fig.add_annotation(x=chatimemedian, y=0.8, yref='paper',
            text="Mediaan tijd om op te laden",
            showarrow=True, ay=-80)

fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)

#titels en astitels
fig.update_layout(title='Oplaadtijd en connectietijd (zonder uitschieters) met kansdichtheidbenadering')
fig.update_xaxes(title='Tijd in uren')
fig.update_yaxes(title='Dichtheid')

fig.show()


# In[ ]:




