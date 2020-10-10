function scatter(vertices, size, color, texture = "") {
    let geometry = new THREE.BufferGeometry();
    let settings = {
        size: size,
        sizeAttenuation: false,
        alphaTest: 0.5,
        transparent: true
    };
    if (texture != "") {
        console.log(texture);
        settings["map"] = new THREE.TextureLoader().load(texture);
    }
    geometry.addAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    material = new THREE.PointsMaterial(settings);
    material.color.set(color);
    return new THREE.Points(geometry, material);
}

function boxEdge(dims, pos, rots, edgewidth, color) {
    let boxes = [];
    for (var i = 0; i < dims.length; ++i) {
        let cube = new THREE.BoxGeometry(dims[i][0], dims[i][1], dims[i][2]);
        var edgeGeo = new THREE.EdgesGeometry(cube);
        let material = new THREE.LineBasicMaterial({
            color: color,
            linewidth: edgewidth
        });
        let edges = new THREE.LineSegments(edgeGeo, material);
        edges.position.set(pos[i][0], pos[i][1], pos[i][2]);
        edges.rotation.set(rots[i][0], rots[i][1], rots[i][2]);
        boxes.push(edges);
    }
    return boxes;
}

function boxEdgeWithLabel(dims, locs, rots, edgewidth, color, labels, lcolor) {
    let boxes = [];
    for (var i = 0; i < dims.length; ++i) {
        let cube = new THREE.BoxGeometry(dims[i][0], dims[i][1], dims[i][2]);
        var edgeGeo = new THREE.EdgesGeometry(cube);
        let material = new THREE.LineBasicMaterial({
            color: color,
            linewidth: edgewidth
        });
        let edges = new THREE.LineSegments(edgeGeo, material);
        edges.position.set(locs[i][0], locs[i][1], locs[i][2]);
        edges.rotation.set(rots[i][0], rots[i][1], rots[i][2]);

        var labelDiv = document.createElement( 'div' );
        labelDiv.className = 'label';
        labelDiv.textContent = labels[i];
        labelDiv.style.color = lcolor;
        // labelDiv.style.marginTop = '-1em';
        labelDiv.style.fontSize = "150%";
        var labelObj = new THREE.CSS2DObject( labelDiv );
        labelObj.position.set( 0, 0, dims[i][2]/2+locs[i][2] );
        edges.add(labelObj);
        boxes.push(edges);
    }
    return boxes;
}

function circle(locs, color, high, size, edgewidth)
{
    let boxes = [];
    for (var j = 0; j < locs.length; j++)
    {
    for (var i = 0; i < locs[j].length; ++i) {
        let cube = new THREE.BoxGeometry(size, size, size);
        var edgeGeo = new THREE.EdgesGeometry(cube);
        let material = new THREE.LineBasicMaterial({
            color: color,
            linewidth: edgewidth
        });
        let edges = new THREE.LineSegments(edgeGeo, material);
        edges.position.set(locs[j][i][0], locs[j][i][1], high);
        edges.rotation.set(0.0, 0.0, 0.0);
        boxes.push(edges);
    }
    }
    return boxes;
}

function draw_line(locs, color, high, edgewidth)
{
    let lines = [];
    for(var i = 0; i < locs.length; ++i)
    {
        for(var j = 0; j < locs[i].length-1; ++j)
        {
            let geometry = new THREE.Geometry();
            geometry.vertices.push(new THREE.Vector3(locs[i][j][0], locs[i][j][1], high));
            geometry.vertices.push(new THREE.Vector3(locs[i][j+1][0], locs[i][j+1][1], high));
            let material = new THREE.LineBasicMaterial({
                color: color,
                linewidth: edgewidth
            });
            let line = new THREE.Line(geometry, material);
            lines.push(line);
        }
    }
    return lines
}

function draw_line_v2(velocity, color, edgewidth, showBox, names_trans, dt_flag)
{
    let lines = [];
    var final_color;
    for(var i = 0; i < velocity.length; ++i)
    {
        if(dt_flag == true && names_trans[i] != "car")
        {
            continue;
        }

        final_color = color;
        let geometry = new THREE.Geometry();
        geometry.vertices.push(new THREE.Vector3(velocity[i][0][0], velocity[i][0][1], velocity[i][0][2]));
        if(showBox)
        {
            geometry.vertices.push(new THREE.Vector3(velocity[i][1][0], velocity[i][1][1], velocity[i][1][2]));
        }
        else
        {
            geometry.vertices.push(new THREE.Vector3(velocity[i][2][0], velocity[i][2][1], velocity[i][2][2]));
        }
        let material = new THREE.LineBasicMaterial({
           color: final_color,
           linewidth: edgewidth
        });
        let line = new THREE.Line(geometry, material);
        lines.push(line);
    }
    return lines
}

function drawCircle(distance, edgewidth, circle_color)
{
    let circles = [];
    var segments = 64;
    for (var i = 0; i < distance.length; i++)
    {
        let material = new THREE.LineBasicMaterial({
            color: circle_color,
            linewidth: edgewidth
        });

        let geometry = new THREE.CircleGeometry(distance[i], segments);
        geometry.vertices.shift();

        circles.push(new THREE.LineLoop(geometry, material))
    }
    return circles;
}

function boxEdgeWithLabelV1(dims, locs, rots, edgewidth, color, labels, names_trans, lcolor, showLabel, gt_flag, ids, type, color_list) {
    let boxes = [];
    var final_color;
    for (var i = 0; i < dims.length; ++i) {
        if(type != "total" && names_trans[i] != type)
        {
            continue;
        }
        let cube = new THREE.BoxGeometry(dims[i][0], dims[i][1], dims[i][2]);
        var edgeGeo = new THREE.EdgesGeometry(cube);
        if(gt_flag)
            final_color = color;
        else
        {
            final_color = color_list[i];
        }
        let material = new THREE.LineBasicMaterial({
            color: final_color,
            linewidth: edgewidth
        });
        let edges = new THREE.LineSegments(edgeGeo, material);
        edges.position.set(locs[i][0], locs[i][1], locs[i][2]);
        edges.rotation.set(rots[i][0], rots[i][1], rots[i][2]);
        let labelObj;
        if(showLabel)
        {
            labelObj = makeTextSprite(ids[i]+"/"+labels[i], {
                fontcolor: lcolor
            });
        }
        else
        {
            labelObj = makeTextSprite("", {
                fontcolor: lcolor
            });
        }
        if(gt_flag)
        {
            labelObj.position.set(0, 0, dims[i][2] / 2);
        }
        else
        {
            labelObj.position.set(dims[i][0]/2, 0, dims[i][2] / 2);
        }
        // labelObj.position.normalize();
        labelObj.scale.set(2, 1, 1.0);
        edges.add(labelObj);
        boxes.push(edges);
    }
    return boxes;
}

function boxEdgeWithLabelV2(dims, locs, rots, edgewidth, color, labels, names_trans, lcolor, showLabel, gt_flag, error, ids, error_color, id_switch, type, ids_Color) {
    let boxes = [];
    var final_color;
    for (var i = 0; i < dims.length; ++i) {
        if(type != "total" && names_trans[i] != type)
        {
            continue;
        }
        let cube = new THREE.BoxGeometry(dims[i][0], dims[i][1], dims[i][2]);
        var edgeGeo = new THREE.EdgesGeometry(cube);
        if(error.length > 0 && error.includes(ids[i]))
        {
            final_color = error_color;
        }
        else
        {
            final_color = color;
        }
        if(gt_flag && id_switch.length > 0 && id_switch.includes(ids[i]))
        {
            this.logger.message(id_switch)
            final_color = ids_Color;
        }
        let material = new THREE.LineBasicMaterial({
            color: final_color,
            linewidth: edgewidth
        });
        let edges = new THREE.LineSegments(edgeGeo, material);
        edges.position.set(locs[i][0], locs[i][1], locs[i][2]);
        edges.rotation.set(rots[i][0], rots[i][1], rots[i][2]);
        let labelObj;
        if(showLabel)
        {
            labelObj = makeTextSprite(ids[i]+"/"+labels[i], {
                fontcolor: lcolor
            });
        }
        else
        {
            labelObj = makeTextSprite("", {
                fontcolor: lcolor
            });
        }
        if(gt_flag)
        {
            labelObj.position.set(0, 0, dims[i][2] / 2);
        }
        else
        {
            labelObj.position.set(dims[i][0]/2, 0, dims[i][2] / 2);
        }
        // labelObj.position.normalize();
        labelObj.scale.set(2, 1, 1.0);
        edges.add(labelObj);
        boxes.push(edges);
    }
    return boxes;
}

function box3D(dims, pos, rots, color, alpha) {
    let boxes = [];
    for (var i = 0; i < dims.length; ++i) {
        let cube = new THREE.BoxGeometry(dims[i][0], dims[i][1], dims[i][2]);
        let material = new THREE.MeshBasicMaterial({
            color: color,
            transparent: alpha != 1.0,
            opacity: alpha
        });
        let box = new THREE.Mesh(cube, material);
        box.position.set(pos[i][0], pos[i][1], pos[i][2]);
        boxes.push(box);
    }
    return boxes;
}

function getKittiInfo(backend, root_path, info_path, callback) {
    backendurl = backend + '/api/readinfo';
    data = {};
    data["root_path"] = root_path;
    data["info_path"] = info_path;
    return $.ajax({
        url: backendurl,
        method: 'POST',
        contentType: "application/json",
        data: JSON.stringify(data),
        success: function (response) {
            return callback(response["results"][0]);
        }
    });
}

function loadKittiDets(backend, det_path, callback) {
    backendurl = backend + '/api/read_detection';
    data = {};
    data["det_path"] = det_path;
    return $.ajax({
        url: backendurl,
        method: 'POST',
        contentType: "application/json",
        data: JSON.stringify(data),
        success: function (response) {
            return callback(response["results"][0]);
        }
    });
}

function getPointCloud(backend, image_idx, with_det, callback) {
    backendurl = backend + '/api/get_pointcloud';
    data = {};
    data["image_idx"] = image_idx;
    data["with_det"] = with_det;
    return $.ajax({
        url: backendurl,
        method: 'POST',
        contentType: "application/json",
        data: JSON.stringify(data),
        success: function (response) {
            return callback(response["results"][0]);
        }
    });
}

function str2buffer(str) {
    var buf = new ArrayBuffer(str.length); // 2 bytes for each char
    var bufView = new Uint8Array(buf);
    for (var i = 0, strLen = str.length; i < strLen; i++) {
        bufView[i] = str.charCodeAt(i);
    }
    return buf;
}

function choose(choices) {
    var index = Math.floor(Math.random() * choices.length);
    return choices[index];
}

function makeTextSprite(message, opts) {
    var parameters = opts || {};
    var fontface = parameters.fontface || 'Helvetica';
    var fontsize = parameters.fontsize || 70;
    var fontcolor = parameters.fontcolor || 'rgba(0, 1, 0, 1.0)';
    var canvas = document.createElement('canvas');
    var context = canvas.getContext('2d');
    context.font = fontsize + "px " + fontface;
  
    // get size data (height depends only on font size)
    var metrics = context.measureText(message);
    var textWidth = metrics.width;
  
    // text color
    context.fillStyle = fontcolor;
    context.fillText(message, 0, fontsize);
  
    // canvas contents will be used for a texture
    var texture = new THREE.Texture(canvas)
    texture.minFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
  
    var spriteMaterial = new THREE.SpriteMaterial({
        map: texture,
    });
    var sprite = new THREE.Sprite(spriteMaterial);
    // sprite.scale.set(5, 5, 1.0);
    return sprite;
  }
