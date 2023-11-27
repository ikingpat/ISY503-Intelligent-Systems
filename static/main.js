var button, input_field, loader, sentimentText, sentimentText2;

button = document.getElementById("sent-btn");
input_field = document.getElementById("sent-input");
loader = document.getElementById("loader");
sentimentText = document.getElementById("api-res")
sentimentText2 = document.getElementById("api-res-2")

sentimentText.style.display = "none"
sentimentText2.style.display = "none"


button.onclick = () => {
    button.style.display = "none"
    loader.style.display = "block"
    
    fetch("/predict",{
        method: "POST",
        body: JSON.stringify({
            data : input_field.value
        }),
        headers: {
            "Content-type": "application/json; charset=UTF-8"
        }
    })
    .then(response => response.json())
    .then((data) => {
        input_field.value = ""
        loader.style.display = "none";
        button.style.display = "block"

        sentimentText.innerText = data.sentiment;
        sentimentText2.innerText = data.probability
        sentimentText.style.display = "block";
    })

}
