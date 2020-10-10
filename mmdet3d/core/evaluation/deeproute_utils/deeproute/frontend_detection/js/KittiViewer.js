var KittiViewer = function (pointCloud, logger, imageCanvas) {
    this.occlusionPath = "/home/jiamiaoxu/data/code/detection_evaluation/detection_Jiamiao/second/kittiviewer/occlusion_detection.pkl";
    this.pcdPath = "/home/jiamiaoxu/data/code/detection_evaluation/detection_Jiamiao/second/pcd";
    this.detPath = "/home/jiamiaoxu/data/code/detection_evaluation/detection_Jiamiao/second/evaluations/dolphin_res_json";
    this.gtPath = "/home/jiamiaoxu/data/code/detection_evaluation/detection_Jiamiao/second/cone_benchmark/label";
    this.backend = "http://127.0.0.1:16666";
    this.checkpointPath = "/home/jiamiaoxu/data/code/detection_evaluation/perception_evaluation/configs/detection.eval.config";
    this.configPath = "/home/jiamiaoxu/data/code/detection_evaluation/perception_evaluation/configs/detection.eval.config";
    this.evalPath = "/home/jiamiaoxu/data/code/detection_evaluation/detection_Jiamiao/second/evaluations/eval_detection";
    this.drawDet = true;
    this.tracking = false;
    this.imageIndexes = [];
    this.imageIndex = 0;
    this.gtBoxes = [];
    this.dtBoxes = [];
    this.gtBboxes = [];
    this.dtBboxes = [];
    this.pointCloud = pointCloud;
    this.maxPoints = 150000;
    this.pointVertices = new Float32Array(this.maxPoints * 3);
    this.missingColor = "#e5dd12";
    this.fpColor = "#f90c1a";
    this.labelColor = "#a55d8d";
    this.dtLabelColor = "#50a6c1";
    this.gtBoxColor = "#3cdd04";
    this.dtBoxColor = "#50a6c1";
    this.boxColor="#3cdd04";
    this.logger = logger;
    this.imageCanvas = imageCanvas;
    this.image = '';
    this.removeOccluded = false;
    this.within_80 = false;
    this.within_40 = false;
    this.within_20 = false;
    this.enableInt16 = true;
    this.int16Factor = 100;
    this.removeOutside = true;
    this.timeStamps = [];
    this.timeStamp = 0;
    this.flag = null;
    this.showError = false;
    this.focusRange = "full";
    this.showLabel = true;
    this.PCformat = false;
    this.showOccluded = false;
    this.type = "car";
    this.circles = [];
    this.showDistance = true;
};

KittiViewer.prototype = {
    readCookies : function(){
        if (CookiesKitti.get("kittiviewer_occlusionPath")){
            this.occlusionPath = CookiesKitti.get("kittiviewer_occlusionPath");
        }
        if (CookiesKitti.get("kittiviewer_backend")){
            this.backend = CookiesKitti.get("kittiviewer_backend");
        }
        if (CookiesKitti.get("kittiviewer_pcdPath")){
            this.pcdPath = CookiesKitti.get("kittiviewer_pcdPath");
        }
        if (CookiesKitti.get("kittiviewer_gtPath")){
            this.gtPath = CookiesKitti.get("kittiviewer_gtPath");
        }
        if (CookiesKitti.get("kittiviewer_detPath")){
            this.detPath = CookiesKitti.get("kittiviewer_detPath");
        }
        if (CookiesKitti.get("kittiviewer_evalPath")){
            this.evalPath = CookiesKitti.get("kittiviewer_evalPath");
        }
        if (CookiesKitti.get("kittiviewer_focusRange")){
            this.focusRange = CookiesKitti.get("kittiviewer_focusRange");
        }
        if (CookiesKitti.get("kittiviewer_configPath")){
            this.configPath = CookiesKitti.get("kittiviewer_configPath");
        }
        if (CookiesKitti.get("kittiviewer_type")){
            this.type = CookiesKitti.get("kittiviewer_type");
        }
        /*if (CookiesKitti.get("kittiviewer_checkpointPath")){
            this.checkpointPath = CookiesKitti.get("kittiviewer_checkpointPath");
        }
        if (CookiesKitti.get("kittiviewer_configPath")){
            this.configPath = CookiesKitti.get("kittiviewer_configPath");
        }
        if (CookiesKitti.get("kittiviewer_infoPath")){
            this.infoPath = CookiesKitti.get("kittiviewer_infoPath");
        }*/
    },
    /*load: function () {
        let self = this;
        let data = {};
        data["root_path"] = this.rootPath;
        data["info_path"] = this.infoPath;
        return $.ajax({
            url: this.addhttp(this.backend) + '/api/readinfo',
            method: 'POST',
            contentType: "application/json",
            data: JSON.stringify(data),
            error: function (jqXHR, exception) {
                self.logger.error("load info fail, please check your backend!");
                console.log("load info fail, please check your backend!");
            },
            success: function (response) {
                let result = response["results"][0];
                self.imageIndexes = [];
                for (var i = 0; i < result["image_indexes"].length; ++i)
                    self.imageIndexes.push(result["image_indexes"][i]);
                self.logger.message("load info success!");
            }
        });
    },*/
    addhttp: function (url) {
        if (!/^https?:\/\//i.test(url)) {
            url = 'http://' + url;
        }
        return url
    },

    load: function () {
        let self = this;
        let data = {};
        data["occlusion_path"] = this.occlusionPath;
        data["gt_path"] = this.gtPath;
        data["det_path"] = this.detPath;
        data["pcd_path"] = this.pcdPath;
        data["eval_path"] = this.evalPath;
        data["removeOccluded"] = this.removeOccluded;
        data["focus_range"] = this.focusRange;
        data["pc_format"] = this.PCformat;
        data["config_path"] = this.configPath;
        data["type"] = this.type;
        //data["within_80"]=this.within_80;
        //data["within_40"]=this.within_40;
        //data["within_20"]=this.within_20;
        $(".imgidx")[0].value = 0;
        return $.ajax({
            url: this.addhttp(this.backend) + '/api/read_all',
            method: 'POST',
            contentType: "application/json",
            data: JSON.stringify(data),
            error: function (jqXHR, exception) {
                self.logger.error("load detection fail!");
                console.log("load detection fail!");
            },
            success: function (response) {
                let result = response["results"][0];
                self.timeStamps = [];
                for (var i = 0; i < result["timeStamps"].length; ++i)
                     self.timeStamps.push(result["timeStamps"][i]);
                self.timeStamp = result["timeStamps"][0];
                self.imageIndex = 0;

                self.imageIndexes = [];
                for (var i = 0; i < result["error_image_indexes"].length; ++i)
                    self.imageIndexes.push(result["error_image_indexes"][i]);
                self.logger.message("load detection success!");
                $(".imgidx")[0].value = this.imageIndex.toString();
            }
        });
    },
    autoplay: function () {
        if(this.flag)
        {
            this.logger.message("Message: pause video")
            clearInterval(this.flag);
            this.flag = null;
        }
        else
        {
            this.logger.message("Message: autoplay video")
            this.flag = setInterval(show, 150);
        }
    },

    plot: function () {
        return this._plot(this.timeStamp);
    },
    plot_v2: function () {
         if(this.showError)
        {
            if(this.imageIndex < this.imageIndexes.length)
            {
                this.timeStamp = this.imageIndexes[this.imageIndex-1]
            }
        }
        else
        {
            if(this.imageIndex < this.timeStamps.length)
            {
                this.timeStamp = this.timeStamps[this.imageIndex-1];
            }
        }
        return this._plot(this.timeStamp);
    },
    next: function () {
        if(this.showError)
        {
            if(this.imageIndex < this.imageIndexes.length)
            {
                this.imageIndex = this.imageIndex + 1;
                $(".imgidx")[0].value = this.imageIndex.toString()
                this.timeStamp = this.imageIndexes[this.imageIndex-1]
                $(".timestampidx")[0].value = this.timeStamp.toString()
                return this.plot();
            }
        }
        else
        {
            if(this.imageIndex < this.timeStamps.length){
                this.imageIndex = this.imageIndex + 1;
                $(".imgidx")[0].value = this.imageIndex.toString()
                this.timeStamp = this.timeStamps[this.imageIndex-1];
                $(".timestampidx")[0].value = this.timeStamp.toString()
                return this.plot();
            }
        }
    },
    prev: function () {
        if(this.showError)
        {
            if(this.imageIndex > 1 && this.imageIndexes.length > 0)
            {
                this.imageIndex = this.imageIndex - 1;
                $(".imgidx")[0].value = this.imageIndex.toString()
                this.timeStamp = this.imageIndexes[this.imageIndex-1]
                $(".timestampidx")[0].value = this.timeStamp.toString()
                return this.plot();
            }
        }
        else
        {
            if(this.imageIndex > 1 && this.timeStamps.length > 0){
            this.imageIndex = this.imageIndex - 1;
            this.timeStamp = this.timeStamps[this.imageIndex-1];
            $(".timestampidx")[0].value = this.timeStamp.toString()
            return this.plot();
            }
        }
    },
    clear: function(){
        for (var i = 0; i < this.gtBoxes.length; ++i) {
            for (var j = this.gtBoxes[i].children.length - 1; j >= 0; j--) {
                this.gtBoxes[i].remove(this.gtBoxes[i].children[j]);
            }
            scene.remove(this.gtBoxes[i]);
            this.gtBoxes[i].geometry.dispose();
            this.gtBoxes[i].material.dispose();
        }
        this.gtBoxes = [];
        for (var i = 0; i < this.dtBoxes.length; ++i) {
            for (var j = this.dtBoxes[i].children.length - 1; j >= 0; j--) {
                this.dtBoxes[i].remove(this.dtBoxes[i].children[j]);
            }
            scene.remove(this.dtBoxes[i]);
            this.dtBoxes[i].geometry.dispose();
            this.dtBoxes[i].material.dispose();
        }
        for (var i = 0; i < this.circles.length; ++i) {
            for (var j = this.circles[i].children.length - 1; j >= 0; j--) {
                this.circles[i].remove(this.circles[i].children[j]);
            }
            scene.remove(this.circles[i]);
            this.circles[i].geometry.dispose();
            this.circles[i].material.dispose();
        }
        this.circles = [];

        this.gtBoxes = [];
        this.dtBoxes = [];
        this.gtBboxes = [];
        this.dtBboxes = [];

        // this.image = '';
    },
    _plot: function (timeStamp) {
        console.log(this.timeStamps.length);
        if (this.timeStamps.length != 0 && this.timeStamps.includes(timeStamp)) {
            let data = {};
            data["timeStamp"] = this.timeStamp;
            data["enable_int16"] = this.enableInt16;
            data["int16_factor"] = this.int16Factor;
            //data["remove_outside"] = this.removeOutside;
            let self = this;
           var ajax1 = $.ajax({
                url: this.addhttp(this.backend) + '/api/get_pointcloud',
                method: 'POST',
                contentType: "application/json",
                data: JSON.stringify(data),
                error: function (jqXHR, exception) {
                    self.logger.error("get point cloud fail!!");
                    console.log("get point cloud fail!!");
                },
                success: function (response) {
                    self.clear();
                    response = response["results"][0];
                    var points_buf = str2buffer(atob(response["pointcloud"]));
                    var points;
                    if (self.enableInt16){
                        var points = new Int16Array(points_buf);
                    }
                    else{
                        var points = new Float32Array(points_buf);
                    }

                    var gt_loc = response["gt_loc"];
                    var gt_dims = response["gt_dims"];
                    var gt_yaws = response["gt_yaws"];
                    var gt_names = response["gt_names"];
                    var gt_occlusion = response["gt_occlusion"];
                    var gt_names_trans = response["gt_name_trans"];
                    var dt_loc = response["dt_loc"];
                    var dt_dims = response["dt_dims"];
                    var dt_yaws = response["dt_yaws"];
                    var dt_names = response["dt_names"];
                    var dt_occlusion = response["dt_occlusion"];
                    var dt_names_trans = response["dt_name_trans"];

                    var numFeatures = response["num_features"];

                    var missing_boxes = response["missing_boxes"];
                    var fp_boxes = response["fp_boxes"];

                    var distance = response["distance"];

                    if(self.showDistance)
                    {
                        self.circles = drawCircle(distance, 0.5, self.gtBoxColor);
                        for (var i = 0; i < self.circles.length; ++i) {
                                scene.add(self.circles[i]);
                        }
                    }

                    if(self.showError)
                    {
                        self.gtBoxes = boxEdgeWithLabelV2(missing_boxes, self.missingColor, gt_dims, gt_loc,
                        gt_yaws, 2, self.gtBoxColor, gt_names, self.labelColor, self.showLabel, gt_occlusion, self.showOccluded, true,
                        gt_names_trans, self.type);
                        for (var i = 0; i < self.gtBoxes.length; ++i) {
                            scene.add(self.gtBoxes[i]);
                        }
                        if(self.drawDet)
                        {
                            self.dtBoxes = boxEdgeWithLabelV2(fp_boxes, self.fpColor, dt_dims, dt_loc, dt_yaws, 2,
                            self.dtBoxColor, dt_names, self.labelColor, self.showLabel, dt_occlusion, self.showOccluded, false,
                            dt_names_trans, self.type);
                            for (var i = 0; i < self.dtBoxes.length; ++i) {
                                scene.add(self.dtBoxes[i]);
                            }
                        }
                    }
                    else
                    {
                        self.gtBoxes = boxEdgeWithLabelV1(gt_dims, gt_loc,
                        gt_yaws, 2, self.gtBoxColor, gt_names, self.labelColor, self.showLabel, gt_occlusion, self.showOccluded, true,
                        gt_names_trans, self.type);
                        for (var i = 0; i < self.gtBoxes.length; ++i) {
                            scene.add(self.gtBoxes[i]);
                        }
                        if(self.drawDet)
                        {
                            self.dtBoxes = boxEdgeWithLabelV1(dt_dims, dt_loc, dt_yaws, 2,
                            self.dtBoxColor, dt_names, self.labelColor, self.showLabel, dt_occlusion, self.showOccluded, false,
                            dt_names_trans, self.type);
                            for (var i = 0; i < self.dtBoxes.length; ++i) {
                                scene.add(self.dtBoxes[i]);
                            }
                        }
                    }

                    for (var i = 0; i < Math.min(points.length / numFeatures, self.maxPoints); i++) {
                        for (var j = 0; j < numFeatures; ++j){
                            self.pointCloud.geometry.attributes.position.array[i * 3 + j] = points[
                                i * numFeatures + j];
                        }
                    }
                    if (self.enableInt16){
                        for (var i = 0; i < self.pointCloud.geometry.attributes.position.array.length; i++) {
                            self.pointCloud.geometry.attributes.position.array[i] /=self.int16Factor;
                        }    
                    }
                    self.pointCloud.geometry.setDrawRange(0, Math.min(points.length / numFeatures,
                        self.maxPoints));
                    self.pointCloud.geometry.attributes.position.needsUpdate = true;
                    self.pointCloud.geometry.computeBoundingSphere();
                }
            });
            /*
            var ajax2 = $.ajax({
                url: this.addhttp(this.backend) + '/api/get_image',
                method: 'POST',
                contentType: "application/json",
                data: JSON.stringify(data),
                error: function (jqXHR, exception) {
                    self.logger.error("get image fail!!");
                    console.log("get image fail!!");
                },
                success: function (response) {
                    response = response["results"][0];
                    self.image = response["image_b64"];
                }
            });

            $.when(ajax1, ajax2).done(function(){
                // draw image, bbox
                self.drawImage();
            });*/
        } else {
            if (this.imageIndexes.length == 0){
                this.logger.error("image indexes isn't load, please click load button!");
                console.log("image indexes isn't load, please click load button!!");
            }else{
                this.logger.error("out of range!");
                console.log("out of range!");
            }
        }
    },

    drawImage : function(){
        if (this.image === ''){
            console.log("??????");
            return;
        }
        let self = this;
        var image = new Image();
        image.onload = function() {
            let aspect = image.width / image.height;
            let w = self.imageCanvas.width;
            self.imageCanvas.height = w / aspect;
            let h = self.imageCanvas.height;
            let ctx = self.imageCanvas.getContext("2d");
            console.log("draw image");
            ctx.drawImage(image, 0, 0, w, h);
            let x1, y1, x2, y2;
            /*for (var i = 0; i < self.gtBboxes.length; ++i){
                ctx.beginPath();
                x1 = self.gtBboxes[i][0] * w;
                y1 = self.gtBboxes[i][1] * h;
                x2 = self.gtBboxes[i][2] * w;
                y2 = self.gtBboxes[i][3] * h;
                ctx.rect(x1, y1, x2 - x1, y2 - y1);
                ctx.lineWidth = 1;
                ctx.strokeStyle = "green";
                ctx.stroke();    
            }
            for (var i = 0; i < self.dtBboxes.length; ++i){
                ctx.beginPath();
                x1 = self.dtBboxes[i][0] * w;
                y1 = self.dtBboxes[i][1] * h;
                x2 = self.dtBboxes[i][2] * w;
                y2 = self.dtBboxes[i][3] * h;
                ctx.rect(x1, y1, x2 - x1, y2 - y1);
                ctx.lineWidth = 1;
                ctx.strokeStyle = "blue";
                ctx.stroke();    
            }*/
        };
        image.src = this.image;

    },
    saveAsImage: function(renderer) {
        var imgData, imgNode;
        try {
            var strMime = "image/jpeg";
            var strDownloadMime = "image/octet-stream";
            imgData = renderer.domElement.toDataURL(strMime);
            this.saveFile(imgData.replace(strMime, strDownloadMime), `pc_${this.imageIndex}.jpg`);
        } catch (e) {
            console.log(e);
            return;
        }
    },
    saveFile : function (strData, filename) {
        var link = document.createElement('a');
        if (typeof link.download === 'string') {
            document.body.appendChild(link); //Firefox requires the link to be in the body
            link.download = filename;
            link.href = strData;
            link.click();
            document.body.removeChild(link); //remove the link when done
        } else {
            location.replace(uri);
        }
    }

}
