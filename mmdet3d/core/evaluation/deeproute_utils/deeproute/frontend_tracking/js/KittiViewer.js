var KittiViewer = function (pointCloud, logger, imageCanvas) {
    this.pcdPath = "/home/jiamiaoxu/data/our_data/lidar/velodyne/20190506_longhua_rain_003";
    this.detPath = "/home/jiamiaoxu/data/code/detection_evaluation/tracking_Jiamiao/evaluations/results";
    this.gtPath = "/home/jiamiaoxu/data/code/detection_evaluation/tracking_Jiamiao/groundtruth";
    this.backend = "http://127.0.0.1:16666";
    this.checkpointPath = "/home/xuanlei/Public/detection_evaluation/second/out/voxelnet-102090.tckpt";
    this.configPath = "/home/jiamiaoxu/data/code/detection_evaluation/tracking_Jiamiao/configs/deeproute.eval.config";
    this.evalPath = "/home/jiamiaoxu/data/code/detection_evaluation/tracking_Jiamiao/evaluations/eval/";
    this.poseConfigFile = "/home/jiamiaoxu/data/code/detection_evaluation/tracking_Jiamiao/configs/pose.csv";
    this.lidarConfigFile = "/home/jiamiaoxu/data/code/detection_evaluation/tracking_Jiamiao/configs/lidars_mkz.cfg";
    this.GtVelocityFile = "/home/jiamiaoxu/data/code/detection_evaluation/tracking_Jiamiao/kittiviewer/gt_velocity_confidence.pkl";
    this.drawDet = true;
    this.tracking = false;
    this.imageIndexes = [];
    this.imageIndex = 0;
    this.gtBoxes = [];
    this.dtBoxes = [];
    this.gtBboxes = [];
    this.dtBboxes = [];
    this.gt_trajector_boxes = [];
    this.dt_trajector_boxes = [];
    this.gt_lines = [];
    this.dt_lines = [];
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
    this.idsColor = "#d904fc"
    this.dt_curri_color = ["#e5dd12", "#f90c1a", "#a55d8d", "#50a6c1", "#3cdd04", "#d904fc"]
    this.logger = logger;
    this.imageCanvas = imageCanvas;
    this.image = '';
    this.removeOccluded = true;
    this.within_80 = false;
    this.within_40 = false;
    this.within_20 = false;
    this.enableInt16 = true;
    this.int16Factor = 100;
    this.removeOutside = true;
    this.timeStamps = [];
    this.timeStamp = 0;
    this.flag = null;
    this.showIDS = false;
    this.showError = false;
    this.focusRange = "full";
    this.showLabel = true;
    this.showVelocity = false;
    this.showVelocityBox = false;
    this.PCformat = false;
    //this.sequence = "0000";
    this.type = "car";
    this.showMode = true;
    this.circles = [];
    this.showDistance = true;
    //this.showOccluded = false;
};

KittiViewer.prototype = {
    readCookies : function(){
//        if (CookiesKitti.get("kittiviewer_occlusionPath")){
//            this.occlusionPath = CookiesKitti.get("kittiviewer_occlusionPath");
//        }
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
        if (CookiesKitti.get("kittiviewer_poseConfigFile")){
            this.poseConfigFile = CookiesKitti.get("kittiviewer_poseConfigFile");
        }
        if (CookiesKitti.get("kittiviewer_lidarConfigFile")){
            this.lidarConfigFile = CookiesKitti.get("kittiviewer_lidarConfigFile");
        }
        if (CookiesKitti.get("kittiviewer_GtVelocityFile")){
            this.GtVelocityFile = CookiesKitti.get("kittiviewer_GtVelocityFile");
        }
        /*if (CookiesKitti.get("kittiviewer_sequence")){
            this.sequence = CookiesKitti.get("kittiviewer_sequence");
        }*/
        if (CookiesKitti.get("kittiviewer_type")){
            this.type = CookiesKitti.get("kittiviewer_type");
        }
        if (CookiesKitti.get("kittiviewer_focusRange")){
            this.focusRange = CookiesKitti.get("kittiviewer_focusRange");
        }
        if (CookiesKitti.get("kittiviewer_configPath")){
            this.configPath = CookiesKitti.get("kittiviewer_configPath");
        }
        /*if (CookiesKitti.get("kittiviewer_infoPath")){
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
        data["gt_path"] = this.gtPath;
        data["det_path"] = this.detPath;
        data["pcd_path"] = this.pcdPath;
        data["eval_path"] = this.evalPath;
        data["pc_format"] = this.PCformat;
        data["removeOccluded"] = this.removeOccluded;
        data["type"] = this.type;
        data["focus_range"] = this.focusRange;
        //data["sequence"] = this.sequence;
        data["config_path"] = this.configPath;
        data["show_mode"] = this.showMode;
        data["GtVelocityFile"] = this.GtVelocityFile;
        data["pose_config_file"] = this.poseConfigFile;
        data["lidarConfigFile"] = this.lidarConfigFile;
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
                for (var i = 0; i < result["id_switches"].length; ++i)
                    self.imageIndexes.push(result["id_switches"][i]);

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
    /*buildNet: function( ){
        let self = this;
        let data = {};
        data["checkpoint_path"] = this.checkpointPath;
        data["config_path"] = this.configPath;
        return $.ajax({
            url: this.addhttp(this.backend) + '/api/build_network',
            method: 'POST',
            contentType: "application/json",
            data: JSON.stringify(data),
            error: function (jqXHR, exception) {
                self.logger.error("build network fail!");
                console.log("build network fail!");
            },
            success: function (response) {
                self.logger.message("build network success!");
            }
        });
    },
    inference: function( ){
        let self = this;
        let data = {"image_idx": self.imageIndex, "remove_outside": self.removeOutside};
        return $.ajax({
            url: this.addhttp(this.backend) + '/api/inference_by_idx',
            method: 'POST',
            contentType: "application/json",
            data: JSON.stringify(data),
            error: function (jqXHR, exception) {
                self.logger.error("inference fail!");
                console.log("inference fail!");
            },
            success: function (response) {
                response = response["results"][0];
                var locs = response["dt_locs"];
                var dims = response["dt_dims"];
                var rots = response["dt_rots"];
                var scores = response["dt_scores"];
                self.dtBboxes = response["dt_bbox"];
                for (var i = 0; i < self.dtBoxes.length; ++i) {
                    for (var j = self.dtBoxes[i].children.length - 1; j >= 0; j--) {
                        self.dtBoxes[i].remove(self.dtBoxes[i].children[j]);
                    }
                    scene.remove(self.dtBoxes[i]);
                    self.dtBoxes[i].geometry.dispose();
                    self.dtBoxes[i].material.dispose();
                }
                let label_with_score = [];
                for (var i = 0; i < locs.length; ++i) {
                    label_with_score.push("score=" + scores[i].toFixed(2).toString());
                }
                var fp_boxes=response["fp_boxes"];
                self.dtBoxes = boxEdgeWithLabel(fp_boxes,dims, locs, rots, 2, self.fpColor,self.rightColor,
                    label_with_score,self.dtLabelColor);
                for (var i = 0; i < self.dtBoxes.length; ++i) {
                    scene.add(self.dtBoxes[i]);
                }
                self.drawImage();
            }
        });
    },*/
    plot: function () {
        return this._plot(this.timeStamp);
    },
    plot_v2: function () {
         if(this.showIDS)
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
        if(this.showIDS)
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
        if(this.showIDS)
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
        for (var i = 0; i < this.gt_lines.length; ++i) {
            for (var j = this.gt_lines[i].children.length - 1; j >= 0; j--) {
                this.gt_lines[i].remove(this.gt_lines[i].children[j]);
            }
            scene.remove(this.gt_lines[i]);
            this.gt_lines[i].geometry.dispose();
            this.gt_lines[i].material.dispose();
        }
        this.gt_lines = []
        for (var i = 0; i < this.dt_lines.length; ++i) {
            for (var j = this.dt_lines[i].children.length - 1; j >= 0; j--) {
                this.dt_lines[i].remove(this.dt_lines[i].children[j]);
            }
            scene.remove(this.dt_lines[i]);
            this.dt_lines[i].geometry.dispose();
            this.dt_lines[i].material.dispose();
        }
        this.dt_lines = []

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
            data["type"] = this.type;
            data["focus_range"] = this.focusRange;
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
                    var gt_ids = response["gt_ids"];
                    var gt_names_trans = response["gt_name_trans"];
                    var gt_velocity = response["gt_velocity"]

                    var dt_loc = response["dt_loc"];
                    var dt_dims = response["dt_dims"];
                    var dt_yaws = response["dt_yaws"];
                    var dt_names = response["dt_names"];
                    var dt_ids = response["dt_ids"];
                    var dt_names_trans = response["dt_name_trans"];
                    var color_list = response["color_list"];
                    var dt_velocity = response["dt_velocity"]

                    var missing = response["missing"];
                    var fp = response["fp"];
                    var ids = response["ids"];

                    var numFeatures = response["num_features"];

                    var distance = response["distance"];

                    if(self.showDistance)
                    {
                        self.circles = drawCircle(distance, 0.5, self.gtBoxColor);
                        for (var i = 0; i < self.circles.length; ++i) {
                                scene.add(self.circles[i]);
                        }
                    }

                    if(self.showMode)
                    {
                        self.gtBoxes = boxEdgeWithLabelV2(gt_dims, gt_loc, gt_yaws, 2, self.gtBoxColor,
                                    gt_names, gt_names_trans, self.gtBoxColor, self.showLabel, true, missing, gt_ids, self.missingColor, ids, self.type, self.idsColor);
                        for (var i = 0; i < self.gtBoxes.length; ++i) {
                            scene.add(self.gtBoxes[i]);
                        }
                        if(self.showVelocity)
                        {
                            self.gt_lines = draw_line_v2(gt_velocity, self.gtBoxColor, 2, self.showVelocityBox, gt_names_trans, false)
                            for (var i = 0; i < self.gt_lines.length; ++i) {
                                scene.add(self.gt_lines[i]);
                            }
                        }
                        if(self.drawDet)
                        {
                            self.dtBoxes = boxEdgeWithLabelV2(dt_dims, dt_loc, dt_yaws, 2,
                            self.dtBoxColor, dt_names, dt_names_trans, self.dtBoxColor, self.showLabel, false, fp, dt_ids, self.fpColor, ids, self.type, self.idsColor);
                            for (var i = 0; i < self.dtBoxes.length; ++i) {
                                scene.add(self.dtBoxes[i]);
                            }

                            if(self.showVelocity)
                            {
                                self.dt_lines = draw_line_v2(dt_velocity, self.dtBoxColor, 2, self.showVelocityBox, dt_names_trans, true)
                                for (var i = 0; i < self.dt_lines.length; ++i) {
                                    scene.add(self.dt_lines[i]);
                                }
                            }
                        }
                    }
                    else
                    {
                        self.gtBoxes = boxEdgeWithLabelV1(gt_dims, gt_loc, gt_yaws, 2, self.gtBoxColor,
                                    gt_names, gt_names_trans, self.gtBoxColor, self.showLabel, true, gt_ids, self.type, color_list);
                        for (var i = 0; i < self.gtBoxes.length; ++i)
                        {
                            scene.add(self.gtBoxes[i]);
                        }
                        if(self.showVelocity)
                        {
                            self.gt_lines = draw_line_v2(gt_velocity, self.gtBoxColor, 2, self.showVelocityBox, gt_names_trans, false)
                            for (var i = 0; i < self.gt_lines.length; ++i) {
                                scene.add(self.gt_lines[i]);
                            }
                        }
                        if(self.drawDet)
                        {
                            self.dtBoxes = boxEdgeWithLabelV1(dt_dims, dt_loc, dt_yaws, 2,
                            self.dtBoxColor, dt_names, dt_names_trans, self.dtBoxColor, self.showLabel, false, dt_ids, self.type, color_list);
                            for (var i = 0; i < self.dtBoxes.length; ++i) {
                                scene.add(self.dtBoxes[i]);
                            }

                            if(self.showVelocity)
                            {
                                self.dt_lines = draw_line_v2(dt_velocity, self.dtBoxColor, 2, self.showVelocityBox, dt_names_trans, true)
                                for (var i = 0; i < self.dt_lines.length; ++i) {
                                    scene.add(self.dt_lines[i]);
                                }
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