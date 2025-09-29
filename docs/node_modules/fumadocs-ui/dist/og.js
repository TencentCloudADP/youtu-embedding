import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { ImageResponse } from 'next/og';
export function generateOGImage(options) {
    const { title, description, icon, site, primaryColor, primaryTextColor, ...rest } = options;
    return new ImageResponse(generate({
        title,
        description,
        icon,
        site,
        primaryTextColor,
        primaryColor,
    }), {
        width: 1200,
        height: 630,
        ...rest,
    });
}
export function generate({ primaryColor = 'rgba(255,150,255,0.3)', primaryTextColor = 'rgb(255,150,255)', ...props }) {
    return (_jsxs("div", { style: {
            display: 'flex',
            flexDirection: 'column',
            width: '100%',
            height: '100%',
            color: 'white',
            padding: '4rem',
            backgroundColor: '#0c0c0c',
            backgroundImage: `linear-gradient(to top right, ${primaryColor}, transparent)`,
        }, children: [_jsxs("div", { style: {
                    display: 'flex',
                    flexDirection: 'row',
                    alignItems: 'center',
                    gap: '16px',
                    marginBottom: '12px',
                    color: primaryTextColor,
                }, children: [props.icon, _jsx("p", { style: {
                            fontSize: '56px',
                            fontWeight: 600,
                        }, children: props.site })] }), _jsx("p", { style: {
                    fontWeight: 800,
                    fontSize: '82px',
                }, children: props.title }), _jsx("p", { style: {
                    fontSize: '52px',
                    color: 'rgba(240,240,240,0.8)',
                }, children: props.description })] }));
}
