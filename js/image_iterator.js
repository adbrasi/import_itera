import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "import_itera.image_iterator",

    async setup() {
        api.addEventListener("image_iterator.update", (event) => {
            const data = event.detail;
            const nodeId = data.node;
            if (nodeId == null) return;

            const node = app.graph.getNodeById(Number(nodeId));
            if (!node) return;

            const infoWidget = node.widgets?.find((w) => w.name === "info_display");
            if (infoWidget) {
                infoWidget.value = `${data.info}\n${data.next_info}`;
            }

            const indexWidget = node.widgets?.find((w) => w.name === "index");
            if (indexWidget) {
                indexWidget.value = data.next_index;
            }

            node.setDirtyCanvas(true, true);
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "ImageIterator") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const self = this;

            // Custom draw widget for read-only info display
            const infoWidget = {
                type: "custom",
                name: "info_display",
                value: "Awaiting first run...",
                serialize: false,
                computeSize() {
                    return [0, 44];
                },
                draw(ctx, node, widgetWidth, y) {
                    const lines = (this.value || "").split("\n");
                    ctx.save();
                    ctx.font = "12px monospace";
                    ctx.fillStyle = "#cccccc";
                    ctx.textAlign = "left";

                    const margin = 15;
                    for (let i = 0; i < lines.length; i++) {
                        ctx.fillText(lines[i], margin, y + 15 + i * 16);
                    }
                    ctx.restore();
                },
            };
            this.addCustomWidget(infoWidget);

            // Reset button
            this.addWidget("button", "Reset Iterator", "reset", () => {
                api.fetchApi("/image_iterator/reset", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ node_id: String(self.id) }),
                })
                    .then((res) => res.json())
                    .then(() => {
                        const idxW = self.widgets?.find((w) => w.name === "index");
                        if (idxW) {
                            idxW.value = 0;
                        }

                        const infoW = self.widgets?.find(
                            (w) => w.name === "info_display"
                        );
                        if (infoW) {
                            infoW.value = "Reset! Will re-scan on next run.";
                        }

                        self.setDirtyCanvas(true, true);
                    })
                    .catch((err) => {
                        console.error("Image Iterator reset error:", err);
                    });
            });

            // Adjust node size
            const w = Math.max(this.size[0], 320);
            this.setSize([w, this.size[1] + 80]);

            return result;
        };

        // Handle onExecuted for UI text fallback
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            if (message?.text) {
                const infoWidget = this.widgets?.find(
                    (w) => w.name === "info_display"
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
