function getCurrentFrame(){
    eel.get_current_frame(!lv1.checked)(setData)
}

function setData(output){
    if(output[0].localeCompare("space") == 0){
        historyBox.value += " ";
    }
    else{
        historyBox.value += output[0];
    }
    lastSign.value = output[0];
    videoOutput.src = output[2]
}

libBtn.onclick = function(){
    eel.penASLTRlib()()
};
copyBtn.onclick = function(){
    
};
saveTXT.onclick = function(){

};

setInterval(function(){
    getCurrentFrame();
}, 0.2*1000);