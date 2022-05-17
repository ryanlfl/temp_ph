
#############
## Imports ##
#############

from functools import wraps
from simplejson.errors import JSONDecodeError

import base64
  

from flask import redirect, url_for, make_response, request, Blueprint, session, abort
import requests

import os

###########
## Utils ##
###########
def b64Encode(string):
   
    sample_string_bytes = string.encode("ascii")
    
    base64_bytes = base64.b64encode(sample_string_bytes)
    base64_string = base64_bytes.decode("ascii")
    return base64_string

def b64Decode(string):
    base64_bytes = string.encode("ascii")
    
    final_string_bytes = base64.b64decode(base64_bytes)
    final_string = final_string_bytes.decode("ascii")

    return final_string

def getCssContent(filename):
    with open(filename,'r') as fob:
        file_contents = fob.read()
        css_connector = f"<style>{file_contents}</style>"
    return css_connector

def getJsContent(filename):
    with open(filename,'r') as fob:
        file_contents = fob.read()
        js_connector = f"<script>{file_contents}</script>"
    return js_connector


def prettyHtmlMessage(message, message_tag='h1', css_file = None, js_file = None):
    css_content = js_content = ""

    if(css_file):
        css_content = getCssContent(css_file)
    if(js_file):
        js_content = getJsContent(js_file)

    prettyMessage = f"{css_content}<{message_tag}>{message}</{message_tag}>{js_content}"
    return prettyMessage

#####################
## SecureApp Class ##
#####################



class SecureApp:

    def __init__(self, app, security_checkpoint_base_url='https://scaportal.lflogistics.net', isDashApp=True):

        SecureApp.SECURITY_CHECKPOINT_BASE_URL = security_checkpoint_base_url
        SecureApp.SECURITY_CHECKPOINT_EXT_AUTH_ENDPOINT = "/authorizeUser"
        SecureApp.SECURITY_CHECKPOINT_EXT_AUTH_URL = SecureApp.SECURITY_CHECKPOINT_BASE_URL + SecureApp.SECURITY_CHECKPOINT_EXT_AUTH_ENDPOINT
        SecureApp.SECURITY_CHECKPOINT_EXT_AUTH_USERNAME = 'lfl_user'
        SecureApp.SECURITY_CHECKPOINT_EXT_AUTH_PASSWORD = 'e@15@fYXP#DACzMA'
        
        secret_key = b'\xdd\xd3\xbc\xdf\xfb\x0e*\r\xe3^k\xbcA\x18\xbbhD\x7f\xacO\x85\x9a\xb9\xe6\xcf\xeb\xd9\xe62@\xce\xca\xfc\x03\x9c\xf5\xf8\x9a\xb8\xc63\x19\xdb\t,`R\x1d%\x9fV\x88\x99i\xa5\xda\x8dn\x1c\xc1\x89\x0eye\xd3<\xbc\xd9\xd0\xacB|\x97M\xedg\xc3I\xfcN\xcf\xa1q\x11\x00g\xd0\x7f\xc7h\xa5\xe2VI\x8aZ\x01\xda\xac+{\xed\x80\xbf\xfe><\xed\x0b\xe7&\xc0\xd2\x18\xcc\xe4\x83Wl*\xc3\xac`\xa2\xbe\x074O'

        if isDashApp:
            app.server.config['SECRET_KEY'] = secret_key
        else:
            app.config['SECRET_KEY'] = secret_key

        app.server.config['SESSION_COOKIE_NAME']='lflauth'
        self.app = app


    def requires_authorization_dash(self, roles=[]):
        app = self.app
        SecureApp.roles = roles        
        server = app.server

        for view_func in server.view_functions:

            if view_func == (app.config['url_base_pathname']):
                server.view_functions[view_func] = SecureApp.dash_roles_required(server.view_functions[view_func])


    @staticmethod
    def dash_roles_required(func):

        @wraps(func)
        def decorated_view(*args, **kwargs):
           
            token = session.get('token')
            current_app_url = request.base_url
            payload = {"site_roles":SecureApp.roles,'token':token, 'current_app_url':current_app_url}

            try:
                auth = (SecureApp.SECURITY_CHECKPOINT_EXT_AUTH_USERNAME,SecureApp.SECURITY_CHECKPOINT_EXT_AUTH_PASSWORD)
                resp = requests.post(SecureApp.SECURITY_CHECKPOINT_EXT_AUTH_URL,json=payload, auth=auth)
                data = resp.json()
            except requests.exceptions.ConnectionError as e:
                print(f"Error connecting to securityCheckpoint -->{e}")
                return prettyHtmlMessage("Auth Service is Down!", css_file='assets/style.css', js_file='secure_app/static/js/redirect.js')
            except JSONDecodeError as e:
                if(resp.status_code == 401):
                    print(f"Basic auth to hit /authorizeUser is wrong -->{e}")
                    return prettyHtmlMessage("API Basic Auth Issue",css_file='assets/style.css', js_file='secure_app/static/js/redirect.js')
                return prettyHtmlMessage("Json Decode Error",css_file='assets/style.css', js_file='secure_app/static/js/redirect.js')

                
            if(resp.status_code == 200 and data['status'] == 1):
                resp = make_response(func(*args, **kwargs))
                resp.set_cookie('username',data.get('username'),httponly=False,max_age=3600)
                return resp

            elif(resp.status_code == 401):
                if('redirect_url' in data.keys()):
                    encrypted_rdrlk = b64Encode(current_app_url)
                    login_url = data.get('redirect_url') + f'?rdrlk={encrypted_rdrlk}'
                    return redirect(login_url)
                elif('message' in data.keys()):
                    return prettyHtmlMessage( data.get('message'), css_file='assets/style.css', js_file='secure_app/static/js/redirect.js')
                return prettyHtmlMessage( "Not Authenticated! Try logging in" ,css_file='assets/style.css', js_file='secure_app/static/js/redirect.js')

            elif(resp.status_code  == 403):
                if('message' in data.keys()):
                    return prettyHtmlMessage( data.get('message') , css_file='assets/style.css', js_file='secure_app/static/js/redirect.js')
                return prettyHtmlMessage( "You do not have permission to view this page" , css_file='assets/style.css', js_file='secure_app/static/js/redirect.js')

            abort(500)
        return decorated_view


    #Auth For Flask apps
    @staticmethod
    def requires_authorization(roles=[], require_all=False):
        def _roles_required(f):
            @wraps(f)
            def decorated_view(*args, **kwargs):

                token = session.get('token')
                current_app_url = request.base_url
                payload = {"site_roles":roles,'token':token, 'current_app_url':current_app_url}

                try:
                    auth = (SecureApp.SECURITY_CHECKPOINT_EXT_AUTH_USERNAME,SecureApp.SECURITY_CHECKPOINT_EXT_AUTH_PASSWORD)
                    resp = requests.post(SecureApp.SECURITY_CHECKPOINT_EXT_AUTH_URL,json=payload, auth=auth)
                    data = resp.json()
                except requests.exceptions.ConnectionError as e:
                    print(f"Error connecting to securityCheckpoint -->{e}")
                    return prettyHtmlMessage( "Auth Service is Down!" , css_file='assets/style.css', js_file='secure_app/static/js/redirect.js' )
                except JSONDecodeError as e:
                    if(resp.status_code == 401):
                        print(f"Basic auth to hit /authorizeUser is wrong -->{e}")
                        return prettyHtmlMessage("API Basic Auth Issue" , css_file='assets/style.css', js_file='secure_app/static/js/redirect.js') 
                    return prettyHtmlMessage( "Json Decode Error" , css_file='assets/style.css', js_file='secure_app/static/js/redirect.js')

                  
                if(resp.status_code == 200 and data['status'] == 1):
                    resp = make_response(func(*args, **kwargs))
                    resp.set_cookie('username',data.get('username'),httponly=False,max_age=3600)
                    return resp

                elif(resp.status_code == 401):
                    if('redirect_url' in data.keys()):
                        encrypted_rdrlk = b64Encode(current_app_url)
                        login_url = data.get('redirect_url') + f'?rdrlk={encrypted_rdrlk}'
                        return redirect(login_url)
                    elif('message' in data.keys()):
                        return prettyHtmlMessage( data.get('message') , css_file='assets/style.css', js_file='secure_app/static/js/redirect.js')
                    return prettyHtmlMessage( "Not Authenticated! Try logging in" , css_file='assets/style.css', js_file='secure_app/static/js/redirect.js') 

                elif(resp.status_code  == 403):
                    if('message' in data.keys()):
                        return prettyHtmlMessage( data.get('message') , css_file='assets/style.css', js_file='secure_app/static/js/redirect.js')
                    return prettyHtmlMessage( "You do not have permission to view this page" , css_file='assets/style.css', js_file='secure_app/static/js/redirect.js')

                abort(500)

            return decorated_view
        return _roles_required

