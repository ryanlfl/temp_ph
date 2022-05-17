const REDIRECT_LINK = "http://scaportal.lflogistics.net/portal/index";
const REDIRECT_LINK_NAME = "SCA PORTAL";
const REDIRECT_TIMER = 5;


setTimeout(function(){
    window.location.href = REDIRECT_LINK;
 }, REDIRECT_TIMER * 1000);

var rediect_message = document.createElement("h5");
rediect_message.innerHTML = "You will be re-directed to <a href='" + REDIRECT_LINK + "'>" + REDIRECT_LINK_NAME + "</a> in <span id='redirect_time'>" + REDIRECT_TIMER + "</span> seconds";

rediect_message.style.marginLeft = "0rem";
rediect_message.style.paddingLeft = "0rem";
rediect_message.style.letterSpacing = "0.09rem";

document.body.appendChild(rediect_message);

setInterval(function(){
    redirect_time = document.getElementById('redirect_time');
    redirect_time.innerHTML = parseInt(redirect_time.innerHTML) - 1;

},1000);