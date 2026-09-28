import rasterio
from rasterio.plot import show

# Open and display the GeoTIFF
with rasterio.open("satellite_image.tif") as src:
    show(src)