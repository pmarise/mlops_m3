document.getElementById('inferButton').addEventListener('click', function() {  
    const resultDiv = document.getElementById('result');  
  
    fetch('/infer', {  
        method: 'POST',  
        headers: {  
            'Content-Type': 'application/json'  
        }  
    })  
    .then(response => response.json())  
    .then(data => {  
        if (data.error) {  
            resultDiv.textContent = `Error:: ${data.error}`;  
        } else {  
            resultDiv.textContent = `Predicted class: ${data.result}`;  
        }  
    })  
    .catch(error => {  
        resultDiv.textContent = `Error:: ${error.message}`;  
    });  
});  