from flask import Flask, session, render_template, redirect, url_for, request
from arcgis.gis import GIS
import secrets
import urllib.parse
from flask_bootstrap import Bootstrap5
from datetime import datetime
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField
from wtforms.validators import DataRequired, Length, Optional
from flask_wtf.csrf import CSRFProtect
from wtforms.fields import DateField
from gis_inventory import items_search
from wtforms import StringField
from wtforms.fields import DateField
import os
import sys


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


app = Flask(
    __name__,
    template_folder=resource_path("templates"),
    static_folder=resource_path("static"),
)
# Registering the extensions
bootstrap = Bootstrap5(app)
csrf = CSRFProtect(app)


class TokenForm(FlaskForm):
    url = StringField(
        "ArcGIS Online or Portal URL", validators=[DataRequired(), Length(max=255)]
    )
    username = StringField("Username", validators=[DataRequired(), Length(max=255)])
    password = PasswordField("Password", validators=[DataRequired(), Length(max=255)])
    submit = SubmitField("Submit")


class SearchForm(FlaskForm):
    owner = StringField("Owner")
    group = StringField("Group")
    tag = StringField("Tag")
    content_status = StringField("Content Status")
    created_from = DateField("Created From", default=None, validators=[Optional()])
    created_to = DateField("Created To", default=None, validators=[Optional()])
    modified_from = DateField("Modified From", default=None, validators=[Optional()])
    modified_to = DateField("Modified To", default=None, validators=[Optional()])
    output_path = StringField("Output Path")
    submit = SubmitField("Submit")


app.secret_key = secrets.token_urlsafe(16)
app.config.from_object(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if "token" not in session:
        return redirect(url_for("get_token"))

    # Repopulate the form from the previous search
    query_string = request.args.get("query_string", "/api/data?", type=str)
    unparsed_query = urllib.parse.parse_qs(query_string.replace("/api/data?", ""))
    unparsed_query = {key: value[0] for key, value in unparsed_query.items()}

    # Convert the date strings to datetime objects
    if unparsed_query.get("created_from", None):
        unparsed_query["created_from"] = datetime.strptime(
            unparsed_query["created_from"], "%Y-%m-%d"
        )

    if unparsed_query.get("created_to", None):
        unparsed_query["created_to"] = datetime.strptime(
            unparsed_query["created_to"], "%Y-%m-%d"
        )

    if unparsed_query.get("modified_from", None):
        unparsed_query["modified_from"] = datetime.strptime(
            unparsed_query["modified_from"], "%Y-%m-%d"
        )

    if unparsed_query.get("modified_to", None):
        unparsed_query["modified_to"] = datetime.strptime(
            unparsed_query["modified_to"], "%Y-%m-%d"
        )

    search_form = SearchForm(**unparsed_query)

    if search_form.validate_on_submit():

        search_query_params = {
            "owner": search_form.owner.data,
            "group": search_form.group.data,
            "tag": search_form.tag.data,
            "content_status": search_form.content_status.data,
            "created_from": search_form.created_from.data,
            "created_to": search_form.created_to.data,
            "modified_from": search_form.modified_from.data,
            "modified_to": search_form.modified_to.data,
            "output_path": search_form.output_path.data,
        }

        search_query = urllib.parse.urlencode(
            {
                key: value
                for key, value in search_query_params.items()
                if value is not None
            }
        )
        return redirect(
            url_for(
                "index",
                query_string=f"/api/data?{search_query}",
            ),
        )

    return render_template(
        "index.html",
        query_string=query_string,
        search_form=search_form,
        search_query=request.args.get("search_query", None, type=str),
        url=session["url"],
        current_portal=session.get("url", None),
    )


@app.route("/get_token", methods=["GET", "POST"])
def get_token(message: str = None):

    token_form = TokenForm()
    if token_form.validate_on_submit():
        session["token"] = GIS(
            username=token_form.username.data,
            password=token_form.password.data,
            url=token_form.url.data,
        )._con.token
        session["url"] = token_form.url.data
        return redirect(url_for("index"))

    return render_template(
        "get_token.html",
        message=message,
        token_form=token_form,
        current_portal=session.get("url", None),
    )


@app.route("/logout")
def logout():
    session.pop("token", None)
    session.pop("url", None)
    return redirect(url_for("get_token", message="You have been logged out."))


@app.route("/api/data")
def data():
    append_search_string = request.args.get("search", None, type=str)
    owner = request.args.get("owner", None, type=str)
    group = request.args.get("group", None, type=str)
    tag = request.args.get("tag", None, type=str)
    content_status = request.args.get("content_status", None, type=str)
    created_from = request.args.get("created_from", None, type=str)
    created_to = request.args.get("created_to", None, type=str)
    modified_from = request.args.get("modified_from", None, type=str)
    modified_to = request.args.get("modified_to", None, type=str)
    output_path = request.args.get("output_path", None, type=str)

    # Convert the date strings to datetime objects
    if created_from:
        created_from = datetime.strptime(created_from, "%Y-%m-%d")

    if created_to:
        created_to = datetime.strptime(created_to, "%Y-%m-%d")

    if modified_from:
        modified_from = datetime.strptime(modified_from, "%Y-%m-%d")

    if modified_to:
        modified_to = datetime.strptime(modified_to, "%Y-%m-%d")

    results = items_search(
        gis=GIS(url=session["url"], token=session["token"]),
        append_search_string=append_search_string,
        owner=owner,
        group=group,
        tag=tag,
        content_status=content_status,
        created_from=created_from,
        created_to=created_to,
        modified_from=modified_from,
        modified_to=modified_to,
        output_path=output_path,
    )

    # response
    return {
        "data": results["results"],
    }


if __name__ == "__main__":
    app.run()
