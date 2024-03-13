from arcgis.auth.tools import LazyLoader

try:
    arcpy = LazyLoader("arcpy", strict=True)

    hasArcPy = True
except:
    hasArcPy = False
tempfile = LazyLoader("tempfile")


# ----------------------------------------------------------------------
def scratchFolder():
    if hasArcPy:
        return arcpy.env.scratchFolder
    else:
        return tempfile.gettempdir()


# ----------------------------------------------------------------------
def scratchGDB():
    if hasArcPy:
        return arcpy.env.scratchGDB
    return tempfile.gettempdir()


# ----------------------------------------------------------------------
def json_to_featureclass(json_file, out_fc):
    if hasArcPy:
        return arcpy.JSONToFeatures_conversion(
            in_json_file=json_file, out_features=out_fc
        )[0]
    return None


# ----------------------------------------------------------------------
