function sae = saetrain_isoGauss27Inv(sae, opts, trainfiles)
    for i = 1 : numel(sae.ae);
        disp(['Training AE ' num2str(i) '/' num2str(numel(sae.ae))]);
        layer = i;
        sae.ae{i} = nntrain_isoGauss27Inv(sae, opts, layer, trainfiles);%(sae.ae{i}, x, x, opts);
        %t = nnff(sae.ae{i}, x, x);
        %x = t.a{2};
        %remove bias term
        %x = x(:,2:end);
    end
end
