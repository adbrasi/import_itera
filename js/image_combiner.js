import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "import_itera.image_combiner",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "ImageCombiner") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const infoWidget = {
                type: "custom",
                name: "combine_info",
                value: "Awaiting first combine...",
                serialize: false,
                computeSize() {
                    return [0, 24];
                },
                draw(ctx, node, widgetWidth, y) {
                    ctx.save();
                    ctx.font = "11px monospace";
                    ctx.fillStyle = "#aaccff";
                    ctx.textAlign = "left";
                    ctx.fillText(this.value || "", 15, y + 15);
                    ctx.restore();
                },
            };
            this.addCustomWidget(infoWidget);

            const w = Math.max(this.size[0], 340);
            this.setSize([w, this.size[1] + 40]);

            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            if (message?.text) {
                const infoWidget = this.widgets?.find(
                    (w) => w.name === "combine_info"
                );
                if (infoWidget) {
                    infoWidget.value = Array.isArray(message.text)
                        ? message.text.join("\n")
                        : String(message.text);
                }
                this.setDirtyCanvas(true, true);
            }
        };
    },
});
