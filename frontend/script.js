document.addEventListener('DOMContentLoaded', function() {
    fetch('../revenue_drivers.csv')
        .then(response => response.text())
        .then(data => {
            const tableBody = document.querySelector('#drivers-table tbody');
            const rows = data.trim().split('\n').slice(1);

            rows.forEach(row => {
                const [feature, coefficient] = row.split(',');
                const tr = document.createElement('tr');
                
                const tdFeature = document.createElement('td');
                tdFeature.textContent = feature;
                
                const tdCoefficient = document.createElement('td');
                tdCoefficient.textContent = parseFloat(coefficient).toFixed(4);
                
                tr.appendChild(tdFeature);
                tr.appendChild(tdCoefficient);
                
                tableBody.appendChild(tr);
            });
        })
        .catch(error => console.error('Error loading the CSV file:', error));
});
