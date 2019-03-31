from wtforms import Form, StringField, IntegerField, DateField, PasswordField, SubmitField, BooleanField, SelectField, TextAreaField, RadioField, FieldList, FormField, FileField
from wtforms import validators, ValidationError
from wtforms.widgets import TextArea
from werkzeug import secure_filename

from flask_wtf.file import FileRequired

class CompleteForm(Form):
  headline = StringField('Headline', validators=[validators.DataRequired()])
  body = StringField('News Article', widget=TextArea())
  submit = SubmitField('Submit')