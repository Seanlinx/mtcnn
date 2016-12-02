function writeLabel(image_set)

f=load(sprintf('wider_face_%s.mat', image_set));
fid = fopen(sprintf('%s.txt', image_set), 'a');
for i = 1 : length(f.event_list)
    for j = 1 : length(f.file_list{i})
        folder_name = f.event_list{i};
        file_name = f.file_list{i}{j};
        face_bboxes = f.face_bbx_list{i}{j};
        fprintf(fid, '%s/%s ', folder_name, file_name);
        for k = 1 : size(face_bboxes, 1)
            bbox = face_bboxes(k, :);
            bbox(3) = bbox(1) + bbox(3);
            bbox(4) = bbox(2) + bbox(4);      
            for id = 1:4
                fprintf(fid, '%.2f ', bbox(id));
            end
        end
        fprintf(fid, '\n');
    end
end            
fclose(fid);        
        
