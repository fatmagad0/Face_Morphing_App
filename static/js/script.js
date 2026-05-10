async function startMorphing() {
    const f1 = document.getElementById('face1').files[0];
    const f2 = document.getElementById('face2').files[0];
    const btn = document.getElementById('btn');
    const loader = document.getElementById('loader');
    const output = document.getElementById('outputGif');

    if (!f1 || !f2) {
        alert("Please select both images first!");
        return;
    }

    const formData = new FormData();
    formData.append('face1', f1);
    formData.append('face2', f2);

    btn.disabled = true;
    loader.classList.remove('hidden');
    output.style.display = 'none';

    try {
        const response = await fetch('/morph', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.gif_url) {
            output.src = data.gif_url;
            output.style.display = 'block';
        } else if (data.error) {
            alert("Error: " + data.error);
        }
    } catch (err) {
        alert("Server connection failed!");
    } finally {
        loader.classList.add('hidden');
        btn.disabled = false;
    }
}