import contextily as cx

# Define your area: [West, South, East, North]
bbox = [-71.06, 42.35, -71.05, 42.36] 

# Download directly as a georeferenced image file
cx.bounds2raster(
    *bbox, 
    "satellite_image.tif", 
    source=cx.providers.Esri.WorldImagery, 
    ll=True # Tells the code you are using standard latitude/longitude
)