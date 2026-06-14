window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, context) {
                return {
                    fillColor: feature.properties.fillColor || "#ff0000",
                    color: "transparent",
                    weight: 0,
                    fillOpacity: feature.properties.fillOpacity || 0.2,
                    interactive: feature.properties.interactive || false
                };
            }

            ,
        function1: function(feature, layer, context) {
            var p = feature.properties || {};
            var sinrStr = (p.sinr_db != null) ? (p.sinr_db.toFixed(1) + " dB") : "N/A";
            layer.bindTooltip(
                "Traffic: " + (p.traffic ?? "-") +
                "<br>Status: " + (p.status ?? "-") +
                "<br>SINR: " + sinrStr +
                "<br>Area: " + (p.obstacle ?? "-"), {
                    sticky: true
                }
            );
        }

    }
});