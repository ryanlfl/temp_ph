from ph_dc1_nike_frf_dash_v2 import app

app=app
server=app.server

if(__name__=='__main__'):
	server.run(debug=True,port=8004)
